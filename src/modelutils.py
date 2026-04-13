import os
from contextlib import contextmanager

import torch
import torch.nn as nn
import transformers
from transformers import AutoConfig, AutoModelForCTC
from src.aq import set_layer_precision

WAV2VEC2_LIKE = ("wav2vec2", "hubert")


@contextmanager
def suspend_nn_inits():
    def skip(*args, **kwargs):
        pass

    saved_inits = torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_
    torch.nn.init.kaiming_uniform_ = torch.nn.init.uniform_ = torch.nn.init.normal_ = skip
    try:
        yield
    finally:
        torch.nn.init.kaiming_uniform_, torch.nn.init.uniform_, torch.nn.init.normal_ = saved_inits


def get_model(
    model_path, load_quantized=None, dtype="auto", device_map=None, attn_implementation=None, trust_remote_code=False
):
    if dtype == "auto":
        dtype = (
            AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code).torch_dtype or "auto"
        )
    else:
        dtype = getattr(torch, dtype)

    model_kwargs = {}
    if transformers.__version__ >= "4.38.0":
        model_kwargs["attn_implementation"] = attn_implementation

    model = AutoModelForCTC.from_pretrained(
        pretrained_model_name_or_path=model_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=dtype,
        **model_kwargs,
    )
    if load_quantized:
        print("Initializing model with random weights...")
        print("Loading quantized model ...")
        model = load_quantized_model(model, load_quantized)
    else:
        print("Loading pretrained model ...")

    print("Model loaded successfully ...")
    return model


def get_model_head(model):
    head = torch.nn.ModuleList()
    if model.config.model_type == "hubert":
        head.append(model.lm_head)
    elif model.config.model_type == "wav2vec2":
        head.append(model.lm_head)
    else:
        raise ValueError(f"Unsupported model type {model.config.model_type}")
    return head


def get_lm_logits(inps_, model):
    if model.config.model_type == "hubert":
        hidden_states = inps_.unsqueeze(0)
        lm_logits = model.lm_head(hidden_states)
    elif model.config.model_type == "wav2vec2":
        hidden_states = inps_.unsqueeze(0)
        if model.wav2vec2.adapter is not None:
            hidden_states = model.wav2vec2.adapter(hidden_states)
        lm_logits = model.lm_head(hidden_states)
    else:
        raise ValueError(f"Unsupported model type {model.config.model_type}")
    return lm_logits


def get_layers(model):
    if model.config.model_type == "hubert":
        return model.hubert.encoder.layers
    elif model.config.model_type == "wav2vec2":
        return model.wav2vec2.encoder.layers
    else:
        raise ValueError(f"Unsupported model type {model.config.model_type}")


def find_sublayers(module, layers=(nn.Linear)):
    res = {}
    for name, layer in module.named_modules():
        if isinstance(layer, layers):
            res[name] = layer
    return res


def get_sequential_groups(model):
    if model.config.model_type in WAV2VEC2_LIKE:
        return [
            ["attention.q_proj"],
            ["attention.k_proj"],
            ["attention.v_proj"],
            ["attention.out_proj"],
            ["feed_forward.intermediate_dense"],
            ["feed_forward.output_dense"],
        ]
    else:
        raise ValueError(f"Unsupported model type {model.config.model_type}")


def load_quantized_model(model, load_path):
    """Load quantized model"""
    if model.config.model_type == "hubert":
        for layer_index in range(len(model.hubert.encoder.layers)):
            model.hubert.encoder.layers[layer_index] = torch.load(
                os.path.join(load_path, str(layer_index) + ".pth"),
                map_location=model.hubert.encoder.layers[layer_index].layer_norm.weight.device,
            )
        model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
        return model
    elif model.config.model_type == "wav2vec2":
        for layer_index in range(len(model.wav2vec2.encoder.layers)):
            model.wav2vec2.encoder.layers[layer_index] = torch.load(
                os.path.join(load_path, str(layer_index) + ".pth"),
                map_location=model.wav2vec2.encoder.layers[layer_index].layer_norm.weight.device,
            )
        model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
        return model
    else:
        raise ValueError(f"Unsupported model type {model.config.model_type}")


def load_dequantized_model(model, load_path, prec=0):
    """Load quantized model by dequantizing it"""
    layers = get_layers(model)
    for layer_index in range(len(layers)):
        print("layer", layer_index)
        layer = layers[layer_index]
        quant_layer = torch.load(os.path.join(load_path, str(layer_index) + ".pth"), map_location="cpu")
        set_layer_precision(quant_layer, prec)
        layers[layer_index] = load_linear_layers(layer, quant_layer, model)
    model.load_state_dict(torch.load(os.path.join(load_path, "not_quantized_weights.pt")), strict=False)
    return model


def load_linear_layers(layer, quant_layer, model):
    layer_ident = {}
    for submodule in layer.modules():
        for child_name, child_module in submodule.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)) or "norm" in child_name:
                if child_name in layer_ident:
                    layer_ident[child_name] += 1
                else:
                    layer_ident[child_name] = 1
                quant_count = 0
                for quant_submodule in quant_layer.modules():
                    for quant_child_name, quant_child_module in quant_submodule.named_children():
                        if quant_child_name == child_name:
                            quant_count += 1
                            if quant_count != layer_ident[child_name]:
                                continue
                            if "norm" in child_name and not isinstance(child_module, (nn.Conv2d, nn.Linear)):
                                child_module.weight.data = quant_child_module.weight.data.to(
                                    child_module.weight.dtype
                                ).to(child_module.weight.device)
                            else:
                                child_module.weight.data = (
                                    quant_child_module.quantized_weight(precision=quant_child_module.precision)
                                    .data.to(child_module.weight.dtype)
                                    .to(child_module.weight.device)
                                )
    return layer


def save_not_quantized_weights(model: nn.Module, save_dir: str, prefix: str = ""):
    already_saved_weights = set()
    for layer in get_layers(model):
        for param in layer.parameters():
            already_saved_weights.add(param)
    not_quantized_weights = {
        name: param for name, param in model.named_parameters() if param not in already_saved_weights
    }
    torch.save(not_quantized_weights, os.path.join(save_dir, prefix + "not_quantized_weights.pt"))
