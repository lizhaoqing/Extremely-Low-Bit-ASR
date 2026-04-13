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


def _multi_precision_loss(
    layer: nn.Module,
    inp: torch.Tensor,
    out_target: torch.Tensor,
    num_precisions: int,
    kwargs: dict,
    alpha: float = 0.01,
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
    """
    Compute multi-precision MSE losses for a single batch.

    Returns (total_loss, mse_losses, reg_losses) where:
      mse_losses[i] = MSE(precision_i output, ground truth)  — one per precision, high→low
      reg_losses[i]  = MSE(precision_i+1 output, precision_0 output.detach())
      total_loss     = mse_losses[0] + sum(mse_losses[1:]) + alpha * sum(reg_losses)

    Restores layer precision to -1 (highest) before returning.
    """
    mse_losses: List[torch.Tensor] = []
    reg_losses: List[torch.Tensor] = []
    out_highest = None
    for idx, prec in enumerate(range(num_precisions - 1, -1, -1)):
        set_layer_precision(layer, prec)
        out, *_ = layer(inp, **kwargs)
        mse_losses.append(F.mse_loss(out, out_target))
        if idx == 0:
            out_highest = out.detach()
        else:
            reg_losses.append(F.mse_loss(out, out_highest))
    set_layer_precision(layer, -1)
    total = mse_losses[0] + sum(mse_losses[1:]) + alpha * sum(reg_losses, torch.tensor(0.0))
    return total, mse_losses, reg_losses


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
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs.
    Matches AQLM_f finetune_groupwise: validation-based early stopping, per-precision loss logging.
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

    local_batch_size = args.local_batch_size or 1
    run_validation = bool(valid_inps and valid_outs)

    # Log number of trainable parameters (mirrors AQLM_f diagnostic)
    n_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    print(f"Fine-tuning {n_params:,} parameters")

    optimizer = torch.optim.Adam(
        [p for p in layer.parameters() if p.requires_grad],
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
    )
    scaler = torch.cuda.amp.GradScaler()

    best_loss = float('inf')
    best_state = None
    no_improvement = 0

    def _run_epoch(inps_list, outs_list, train: bool):
        """One pass over inps_list/outs_list. Returns (avg_loss, last_mse_losses, last_reg_losses)."""
        total_loss = 0.0
        n_batches = 0
        last_mse, last_reg = [], []
        device = args.devices[0]
        n = inps_list[0].shape[0]
        indices = torch.randperm(n) if train else torch.arange(n)
        for batch_start in range(0, n, local_batch_size):
            batch_ix = indices[batch_start: batch_start + local_batch_size]
            inp_b = inps_list[0][batch_ix].to(device)
            out_b = outs_list[0][batch_ix].to(device)
            batch_kwargs = {
                k: (v.expand(len(batch_ix), *v.shape[1:]) if isinstance(v, torch.Tensor) and v.shape[0] == 1 and len(batch_ix) > 1 else v)
                for k, v in kwargs.items()
            }
            with torch.cuda.amp.autocast():
                loss, mse_l, reg_l = _multi_precision_loss(
                    layer, inp_b, out_b, args.num_precisions, batch_kwargs
                )
            if train:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            total_loss += loss.item()
            n_batches += 1
            last_mse, last_reg = mse_l, reg_l
        return total_loss / max(n_batches, 1), last_mse, last_reg

    for epoch in range(args.finetune_max_epochs):
        layer.train()
        train_loss, _, _ = _run_epoch(train_inps, train_outs, train=True)

        if run_validation:
            layer.eval()
            with torch.no_grad():
                val_loss, last_mse, last_reg = _run_epoch(valid_inps, valid_outs, train=False)
            monitor_loss = val_loss
        else:
            val_loss = float('nan')
            monitor_loss = train_loss
            # get per-precision breakdown from one train batch for logging only
            layer.eval()
            with torch.no_grad():
                b = min(local_batch_size, train_inps[0].shape[0])
                inp_b = train_inps[0][:b].to(args.devices[0])
                out_b = train_outs[0][:b].to(args.devices[0])
                bk = {k: (v.expand(b, *v.shape[1:]) if isinstance(v, torch.Tensor) and v.shape[0] == 1 and b > 1 else v) for k, v in kwargs.items()}
                with torch.cuda.amp.autocast():
                    _, last_mse, last_reg = _multi_precision_loss(layer, inp_b, out_b, args.num_precisions, bk)
            layer.train()

        if verbose:
            mse_fmt = [f"{l.item():.2e}" for l in last_mse]
            reg_fmt = [f"{l.item():.2e}" for l in last_reg]
            print("-" * 10)
            print(f"epoch={epoch}  train={train_loss:.2e}  val={val_loss:.2e}")
            print(f"  mse_losses(high→low): {mse_fmt}")
            if reg_fmt:
                print(f"  reg_losses:           {reg_fmt}")

        if monitor_loss < best_loss:
            best_loss = monitor_loss
            if args.finetune_keep_best:
                best_state = deepcopy(layer.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.finetune_early_stop:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    if args.finetune_keep_best and best_state is not None:
        layer.load_state_dict(best_state)

    return layer


def _make_parameter_replacement_tables(
    layer: nn.Module, replicas: Sequence[nn.Module], param_names: Sequence[str], parameters: nn.ParameterList
) -> Sequence[List[Sequence[Tuple[nn.Module, str]]]]:
    """
    Prepare auxiliary data structures for quickly copying parameters to replicas for data-parallel training.
    """
    assert len(param_names) == len(parameters)
    assert len(replicas) > 1
    assert replicas[0] is layer

    parameters_by_name = dict(zip(param_names, parameters))

    param_to_name = {param: name for name, param in parameters_by_name.items()}
    param_occurences = defaultdict(list)
    for submodule_name, submodule in layer.named_modules():
        for attr_name, param in submodule.named_parameters(recurse=False):
            if param in param_to_name:
                param_name = param_to_name[param]
                param_occurences[param_name].append((submodule_name, attr_name))
    assert len(param_occurences) == len(parameters), "internal error: not all parameters were found"

    replacement_tables = []
    for replica in replicas:
        replacement_table = list()
        replica_modules_by_name: Dict[str, nn.Module] = dict(replica.named_modules())

        for param_name, master_param in zip(param_names, parameters):
            param_replacements = list()
            for submodule_name, attr_name in param_occurences[param_name]:
                param_replacements.append((replica_modules_by_name[submodule_name], attr_name))
            replacement_table.append(param_replacements)
        replacement_tables.append(replacement_table)
    return replacement_tables


def _compute_mse_on_batch(
    layer: nn.Module, batch_iter: Iterator[Tuple[torch.Tensor, torch.Tensor]], num_precisions, **kwargs
) -> torch.Tensor:
    """
    Compute the activation MSE error between transformer layers
    """
    inps_batch, outs_batch = next(batch_iter)
    inps_batch = inps_batch.to(dtype=torch.float32)
    outs_batch = outs_batch.to(dtype=torch.float32)

    if inps_batch.shape[0] != 1:
        for name, value in list(kwargs.items()):
            if isinstance(value, torch.Tensor) and value.shape[0] == 1:
                if name not in ("attention_mask", "position_ids"):
                    warnings.warn(f"Tiling an unexpected kwarg {name} over batch size; make sure this is valid.")
                repeats = [len(inps_batch)] + [1 for _ in range(value.ndim - 1)]
                kwargs[name] = value.tile(*repeats)
            if isinstance(value, list):
                kwargs[name] = torch.cat(value)

    losses = []
    reg_losses = []
    for idx, prec in enumerate(range(num_precisions-1, -1, -1)):
        set_layer_precision(layer, prec)
        out, *_unused = layer(inps_batch, **kwargs)
        losses.append(F.mse_loss(out, outs_batch))
        if idx == 0:
            outs_prediction_high = out
        else:
            reg_losses.append(F.mse_loss(out, outs_prediction_high.detach()))

    return losses+reg_losses


def _compute_mse_parallel(
    devices: Sequence[torch.device],
    replicas: Sequence[nn.Module],
    parameters_to_replicate: nn.ParameterList,
    replacement_tables: Sequence[List[Sequence[Tuple[nn.Module, str]]]],
    batch_iterators: Sequence[Iterator[Tuple[torch.Tensor, torch.Tensor]]],
    kwargs_by_device: Sequence[Dict[str, Any]],
    num_precisions,
) -> torch.Tensor:
    """Compute MSE in parallel over multiple GPUs, each GPU processes a portion of samples"""
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
    for idx in range(2*num_precisions-1):
        out_losses.append(Gather.apply(devices[0], 0, *(mse[idx].view(1) for mse in losses)).mean())

    return out_losses
