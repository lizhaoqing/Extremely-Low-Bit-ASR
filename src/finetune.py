from __future__ import annotations

import warnings
from argparse import Namespace
from collections import defaultdict
from copy import deepcopy
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel.scatter_gather import Gather

from aq_engine import replace_parameter_
from src.aq import set_layer_precision
from src.utils import iterate_minibatches


@torch.enable_grad()
def finetune_groupwise(
    *,
    layer: nn.Module,
    train_inps: Sequence[torch.Tensor],
    train_outs: Sequence[torch.Tensor],
    args: Namespace,
    valid_inps: Sequence[torch.Tensor] = None,
    valid_outs: Sequence[torch.Tensor] = None,
    verbose: bool = True,
    **kwargs,
) -> nn.Module:
    """
    Fine-tune a quantized layer to minimise layer-wise activation MSE.
    alpha=0.01, gradient accumulation, pre-training eval so "no finetune" is a valid best.
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1
    assert len(train_inps) == len(train_outs) == len(args.devices)
    for i in range(len(args.devices)):
        assert isinstance(train_inps[i], torch.Tensor) and isinstance(train_outs[i], torch.Tensor)
        if not args.offload_activations:
            assert train_inps[i].device == train_outs[i].device == args.devices[i]
        else:
            assert train_inps[i].device == train_outs[i].device == torch.device("cpu")
            assert train_inps[i].is_pinned() and train_outs[i].is_pinned()

    alpha = 0.01

    replicas = kwargs_by_device = replacement_tables = None
    if len(args.devices) > 1:
        replicas = torch.nn.parallel.replicate(layer, args.devices)
        replicas[0] = layer
        kwargs_by_device = [
            {k: (v.to(d, non_blocking=True) if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}
            for d in args.devices
        ]

    diff_params_by_name = {n: p for n, p in layer.named_parameters() if p.requires_grad}
    print(f"differentiable parameters for finetuning:\n  {list(diff_params_by_name.keys())}")
    param_names, diff_params = zip(*diff_params_by_name.items())
    differentiable_parameters = nn.ParameterList(diff_params)
    for p in differentiable_parameters:
        p.grad = torch.zeros_like(p)

    if replicas:
        replacement_tables = _make_parameter_replacement_tables(
            layer, replicas, param_names, differentiable_parameters
        )

    print(f"Fine-tuning {sum(p.numel() for p in differentiable_parameters):,} parameters")

    opt = torch.optim.Adam(
        differentiable_parameters,
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
    )

    assert args.finetune_batch_size % len(args.devices) == 0
    num_samples_per_device = train_inps[0].shape[0]
    local_batch_size = args.local_batch_size or (args.finetune_batch_size // len(args.devices))
    assert args.finetune_batch_size % (local_batch_size * len(args.devices)) == 0
    num_accumulation_steps = args.finetune_batch_size // (local_batch_size * len(args.devices))
    train_batches_per_epoch = num_samples_per_device // local_batch_size

    def make_iters(inps_list, outs_list):
        return [
            iterate_minibatches(inps_list[i], outs_list[i], batch_size=local_batch_size, device=args.devices[i])
            for i in range(len(args.devices))
        ]

    train_iters = make_iters(train_inps, train_outs)

    run_validation = bool(valid_inps and valid_outs)
    if run_validation:
        valid_batches_per_epoch = valid_inps[0].shape[0] // local_batch_size
        valid_iters = make_iters(valid_inps, valid_outs)

    def _run_eval(iters, n_batches):
        loss_num = loss_den = 0
        for _ in range(n_batches):
            if len(args.devices) == 1:
                *losses, = _compute_mse_on_batch(layer, iters[0], args.num_precisions, **kwargs)
            else:
                *losses, = _compute_mse_parallel(
                    args.devices, replicas, differentiable_parameters,
                    replacement_tables, iters, kwargs_by_device, args.num_precisions,
                )
            loss = sum(losses[:args.num_precisions]) + sum(losses[args.num_precisions:]) * alpha
            loss_num += loss.item()
            loss_den += 1
        return loss_num / loss_den, losses

    best_loss = float("inf")
    best_parameters_by_name = None
    worse_count = 0

    # pre-training eval — establishes baseline so "no finetune" is a valid best
    if run_validation:
        layer.eval()
        with torch.no_grad():
            pre_loss, losses = _run_eval(valid_iters, valid_batches_per_epoch)
        mse_fmt = [f"{l.item():.2e}" for l in losses[:args.num_precisions]]
        reg_fmt = [f"{l.item():.2e}" for l in losses[args.num_precisions:]]
        print(f"Evaluation before training. valid loss={pre_loss:.2e}")
        print(f"  mse_losses: {mse_fmt}\n  reg_losses: {reg_fmt}")
        best_loss = pre_loss
        best_parameters_by_name = deepcopy(diff_params_by_name)

    steps_accumulated = 0
    for epoch in range(args.finetune_max_epochs):
        layer.train()
        loss_num = loss_den = 0
        for _ in range(train_batches_per_epoch):
            if len(args.devices) == 1:
                *losses, = _compute_mse_on_batch(layer, train_iters[0], args.num_precisions, **kwargs)
            else:
                *losses, = _compute_mse_parallel(
                    args.devices, replicas, differentiable_parameters,
                    replacement_tables, train_iters, kwargs_by_device, args.num_precisions,
                )
            loss = sum(losses[:args.num_precisions]) + sum(losses[args.num_precisions:]) * alpha
            (loss / num_accumulation_steps).backward()
            steps_accumulated += 1

            if not torch.isfinite(loss):
                raise ValueError(f"Fine-tuning loss is {loss}")

            if steps_accumulated >= num_accumulation_steps:
                opt.step()
                opt.zero_grad()
                steps_accumulated = 0

            loss_num += loss.item()
            loss_den += 1
        train_loss_epoch = loss_num / loss_den

        if run_validation:
            layer.eval()
            with torch.no_grad():
                val_loss_epoch, losses = _run_eval(valid_iters, valid_batches_per_epoch)

        if verbose:
            print("-" * 10)
            print(f"epoch={epoch}  train={train_loss_epoch:.2e}", end="")
            if run_validation:
                mse_fmt = [f"{l.item():.2e}" for l in losses[:args.num_precisions]]
                reg_fmt = [f"{l.item():.2e}" for l in losses[args.num_precisions:]]
                print(f"  val={val_loss_epoch:.2e}")
                print(f"  mse_losses: {mse_fmt}\n  reg_losses: {reg_fmt}")
            else:
                print()

        if run_validation:
            if val_loss_epoch < best_loss:
                print(f"new best loss {val_loss_epoch:.2e} at epoch {epoch}")
                best_loss = val_loss_epoch
                best_parameters_by_name = deepcopy(diff_params_by_name)
                worse_count = 0
            else:
                worse_count += 1
                if worse_count >= args.finetune_early_stop:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

    if run_validation and best_parameters_by_name is not None:
        layer.load_state_dict(best_parameters_by_name, strict=False)

    return layer


def _make_parameter_replacement_tables(
    layer: nn.Module, replicas: Sequence[nn.Module], param_names: Sequence[str], parameters: nn.ParameterList
) -> Sequence[List[Sequence[Tuple[nn.Module, str]]]]:
    assert len(param_names) == len(parameters)
    assert len(replicas) > 1
    assert replicas[0] is layer

    parameters_by_name = dict(zip(param_names, parameters))
    param_to_name = {param: name for name, param in parameters_by_name.items()}
    param_occurences = defaultdict(list)
    for submodule_name, submodule in layer.named_modules():
        for attr_name, param in submodule.named_parameters(recurse=False):
            if param in param_to_name:
                param_occurences[param_to_name[param]].append((submodule_name, attr_name))
    assert len(param_occurences) == len(parameters), "internal error: not all parameters were found"

    replacement_tables = []
    for replica in replicas:
        replacement_table = []
        replica_modules_by_name: Dict[str, nn.Module] = dict(replica.named_modules())
        for param_name, master_param in zip(param_names, parameters):
            param_replacements = []
            for submodule_name, attr_name in param_occurences[param_name]:
                param_replacements.append((replica_modules_by_name[submodule_name], attr_name))
            replacement_table.append(param_replacements)
        replacement_tables.append(replacement_table)
    return replacement_tables


def _compute_mse_on_batch(
    layer: nn.Module,
    batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]],
    num_precisions: int,
    **kwargs,
) -> List[torch.Tensor]:
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1] * (value.ndim - 1)
                kwargs[name] = value.tile(*repeats)
            if isinstance(value, list):
                kwargs[name] = torch.cat(value)

    losses = []
    reg_losses = []
    for idx, prec in enumerate(range(num_precisions - 1, -1, -1)):
        set_layer_precision(layer, prec)
        out, *_unused = layer(inps_batch, **kwargs)
        losses.append(F.mse_loss(out, outs_batch))
        if idx == 0:
            out_highest = out
        else:
            reg_losses.append(F.mse_loss(out, out_highest.detach()))

    return losses + reg_losses


def _compute_mse_parallel(
    devices: Sequence[torch.device],
    replicas: Sequence[nn.Module],
    parameters_to_replicate: nn.ParameterList,
    replacement_tables: Sequence[List[Sequence[Tuple[nn.Module, str]]]],
    batch_iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    kwargs_by_device: Sequence[Dict[str, Any]],
    num_precisions: int,
) -> List[torch.Tensor]:
    replicated_parameters = torch.nn.parallel.replicate(parameters_to_replicate, devices, detach=False)
    funcs_by_replica = [_compute_mse_on_batch for _ in replicas]
    inputs_by_replica = []
    for i in range(len(devices)):
        if i != 0:
            for replacement_param, replacement_table in zip(replicated_parameters[i], replacement_tables[i]):
                for (replica_submodule, attr_name) in replacement_table:
                    replace_parameter_(replica_submodule, attr_name, replacement_param)
        inputs_by_replica.append((replicas[i], batch_iterators[i], num_precisions))

    losses = torch.nn.parallel.parallel_apply(
        funcs_by_replica, inputs_by_replica, kwargs_by_device, devices=devices
    )
    out_losses = []
    for idx in range(2 * num_precisions - 1):
        out_losses.append(Gather.apply(devices[0], 0, *(mse[idx].view(1) for mse in losses)).mean())
    return out_losses
