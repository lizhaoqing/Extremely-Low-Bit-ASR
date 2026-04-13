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
    Fine-tune a module with pre-quantized linear layers so as to minimize MSE between layer-wise inps/outs

    :param layer: a trainable module where linear layers are replaced by QuantizedLinear instances
    :param inps: a list of tensors of input activations, [nsamples_per_device, seq_len, hidden_size]
    :param outs: a list of tensors of previous output activations, [nsamples_per_device, seq_len, hidden_size]
    :param args: quantization hyperparameters from main.py
    :param kwargs: additional keyword arguments to be passed into layer on each forward
    """
    assert isinstance(args.devices, (list, tuple)) and len(args.devices) >= 1, f"Found devices = {args.devices}"
    assert isinstance(train_inps, (list, tuple)) and isinstance(train_inps, (list, tuple))
    assert len(train_inps) == len(train_outs) == len(args.devices)
    for i in range(len(args.devices)):
        assert isinstance(train_inps[i], torch.Tensor) and isinstance(train_outs[i], torch.Tensor)
        if not args.offload_activations:
            assert train_inps[i].device == train_outs[i].device == args.devices[i], (
                train_inps[i].device,
                train_outs[i].device,
                args.devices,
            )
        else:
            assert train_inps[i].device == train_outs[i].device == torch.device("cpu")
            assert train_inps[i].is_pinned() and train_outs[i].is_pinned()

    # Enable gradient checkpointing if available
    if hasattr(layer, 'gradient_checkpointing_enable'):
        layer.gradient_checkpointing_enable()

    # Enable mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Reduce batch size if specified
    local_batch_size = getattr(args, 'local_batch_size', 1)
    if local_batch_size is None:
        local_batch_size = 1

    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        layer.parameters(),
        lr=args.finetune_lr,
        betas=(args.finetune_adam_beta1, args.finetune_adam_beta2),
    )

    best_loss = float('inf')
    best_state = None
    no_improvement = 0

    for epoch in range(args.finetune_max_epochs):
        epoch_loss = 0
        num_batches = 0

        for i in range(len(args.devices)):
            # Clear CUDA cache before processing each device
            torch.cuda.empty_cache()
            n = train_inps[i].shape[0]
            # iterate over a single finite pass (one epoch = one pass over all samples)
            indices = torch.randperm(n)
            for batch_start in range(0, n, local_batch_size):
                batch_ix = indices[batch_start: batch_start + local_batch_size]
                inp_batch = train_inps[i][batch_ix].to(args.devices[i])
                out_batch = train_outs[i][batch_ix].to(args.devices[i])
                # tile kwargs tensors (e.g. attention_mask shape [1,...]) to match batch size
                batch_kwargs = {}
                for k, v in kwargs.items():
                    if isinstance(v, torch.Tensor) and v.shape[0] == 1 and len(batch_ix) > 1:
                        batch_kwargs[k] = v.expand(len(batch_ix), *v.shape[1:])
                    else:
                        batch_kwargs[k] = v
                try:
                    with torch.cuda.amp.autocast():
                        loss = F.mse_loss(layer(inp_batch, **batch_kwargs)[0], out_batch)

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                    epoch_loss += loss.item()
                    num_batches += 1

                except RuntimeError as e:
                    if "out of memory" in str(e):
                        torch.cuda.empty_cache()
                        if local_batch_size > 1:
                            local_batch_size //= 2
                            print(f"Reduced batch size to {local_batch_size} due to OOM")
                            continue
                    raise e

        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('inf')

        if verbose:
            print(f"Epoch {epoch + 1}/{args.finetune_max_epochs}, Loss: {avg_epoch_loss:.6f}")

        # Early stopping check
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_state = deepcopy(layer.state_dict())
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= args.finetune_early_stop:
                if verbose:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    # Restore best model state if requested
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
