import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter, ParameterList
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from model.quant_utils import *


class QuantModule:

    def set_precision(self, precision):
        self.precision_level = precision

    def update_minmax(self, weight):
        weight_transform = weight.data.detach()

        if self.weight_bit > 2:
            if self.max.nelement() == 0 or self.max.data < weight_transform.max().data:
                self.max.data = weight_transform.max().data
        else:
            self.max.data = weight_transform.std() * 2
            if self.max.data > 1:
                self.max.data /= self.max.data
        self.max.clamp_(min=0)

        if self.weight_bit > 2:
            if self.min.nelement() == 0 or self.min.data > weight_transform.min().data:
                self.min.data = weight_transform.min().data
        else:
            self.min.data = -weight_transform.std() * 2
            if self.min.data < -1:
                self.min.data /= -self.min.data
        self.min.clamp_(max=0)


class SwitchLayerNorm(nn.Module, QuantModule):
    """Switchable LayerNorm for multi-precision inference.
    Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, bit_list=[8, 4, 2]):
        super(SwitchLayerNorm, self).__init__()
        self.precision_level = 8
        self.bit_list = bit_list
        self.ln_dict = nn.ModuleDict()
        for i in self.bit_list:
            self.ln_dict[str(i)] = LayerNorm(num_features)

    def forward(self, x):
        x = self.ln_dict[str(self.precision_level)](x)
        return x


class QuantLinear(Module, QuantModule):
    def __init__(self,
                 linear_model,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 shared_layer_num=1,
                 ):
        super(QuantLinear, self).__init__()
        self.linear = linear_model
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.layer_type = 'QuantLinear'
        self.shared_layer_num = shared_layer_num
        self.precision_level = weight_bit
        self.use_scaling = use_scaling

        if weight_alpha:
            self.weight_alpha = weight_alpha
        elif use_scaling:
            self.weight_alpha = nn.ParameterDict()
            self.weight_alpha['1'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['2'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['4'] = Parameter(torch.tensor(1.0))
        else:
            self.weight_alpha = 1

        self.register_buffer('scale_factor', torch.tensor(1., requires_grad=False))
        self.register_buffer('min', torch.tensor(0., requires_grad=False))
        self.register_buffer('max', torch.tensor(0., requires_grad=False))

        if isinstance(linear_model.bias, torch.Tensor):
            self.register_buffer('quant_bias', torch.zeros_like(linear_model.bias))
        else:
            self.register_buffer('quant_bias', None)

        if self.quant_mode == "symmetric":
            if use_scaling:
                self.weight_function = ScaleSymmetricLinearQuant_old.apply
            else:
                self.weight_function = UniformSymmetricLinearQuant.apply
                self.weight_truncate = UniformSymmetricTruncate.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def forward(self, x):

        self.update_minmax(self.linear.weight)

        if self.use_scaling:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.linear.weight,
                        self.weight_alpha[str(self.precision_level)],
                        self.precision_level,
                        self.scale_factor,
                    )
        else:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.linear.weight,
                        self.weight_alpha,
                        self.precision_level,
                        self.scale_factor,
                    )

        x = F.linear(x, weight=self.quant_weight, bias=self.linear.bias)

        return x

    def __repr__(self):
        s = super(QuantLinear, self).__repr__()
        s = "(" + s + " weight_bit={}, quantize_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s


class QuantConv1d(Module, QuantModule):
    def __init__(self,
                 conv1d_model,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 shared_layer_num=1,
                 ):
        super(QuantConv1d, self).__init__()
        self.conv = conv1d_model
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.layer_type = 'QuantConv1d'
        self.shared_layer_num = shared_layer_num
        self.precision_level = weight_bit
        self.use_scaling = use_scaling

        if weight_alpha:
            self.weight_alpha = weight_alpha
        elif use_scaling:
            self.weight_alpha = nn.ParameterDict()
            self.weight_alpha['1'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['2'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['4'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['8'] = Parameter(torch.tensor(1.0))
        else:
            self.weight_alpha = 1

        self.register_buffer('scale_factor', torch.tensor(1., requires_grad=False))
        self.register_buffer('min', torch.tensor(0., requires_grad=False))
        self.register_buffer('max', torch.tensor(0., requires_grad=False))

        if isinstance(conv1d_model.bias, torch.Tensor):
            self.register_buffer('quant_bias', torch.zeros_like(conv1d_model.bias))
        else:
            self.register_buffer('quant_bias', None)

        if self.quant_mode == "symmetric":
            if use_scaling:
                self.weight_function = ScaleSymmetricLinearQuant_old.apply
            else:
                self.weight_function = UniformSymmetricLinearQuant.apply
                self.weight_truncate = UniformSymmetricTruncate.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def forward(self, x):

        self.update_minmax(self.conv.weight)

        if self.use_scaling:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.conv.weight,
                        self.weight_alpha[str(self.precision_level)],
                        self.precision_level,
                        self.scale_factor,
                    )
        else:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.conv.weight,
                        self.weight_alpha,
                        self.precision_level,
                        self.scale_factor,
                    )

        x = F.conv1d(
            x,
            self.quant_weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )

        return x

    def __repr__(self):
        s = super(QuantConv1d, self).__repr__()
        s = "(" + s + " weight_bit={}, quantize_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s


class QuantConv2d(Module, QuantModule):
    def __init__(self,
                 conv2d_model,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 shared_layer_num=1,
                 ):
        super(QuantConv2d, self).__init__()
        self.conv = conv2d_model
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.layer_type = 'QuantConv2d'
        self.shared_layer_num = shared_layer_num
        self.precision_level = weight_bit
        self.use_scaling = use_scaling

        if weight_alpha:
            self.weight_alpha = weight_alpha
        elif use_scaling:
            self.weight_alpha = nn.ParameterDict()
            self.weight_alpha['1'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['2'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['4'] = Parameter(torch.tensor(1.0))
            self.weight_alpha['8'] = Parameter(torch.tensor(1.0))
        else:
            self.weight_alpha = 1

        self.register_buffer('scale_factor', torch.tensor(1., requires_grad=False))
        self.register_buffer('min', torch.tensor(0., requires_grad=False))
        self.register_buffer('max', torch.tensor(0., requires_grad=False))

        if isinstance(conv2d_model.bias, torch.Tensor):
            self.register_buffer('quant_bias', torch.zeros_like(conv2d_model.bias))
        else:
            self.register_buffer('quant_bias', None)

        if self.quant_mode == "symmetric":
            if use_scaling:
                self.weight_function = ScaleSymmetricLinearQuant_old.apply
            else:
                self.weight_function = UniformSymmetricLinearQuant.apply
                self.weight_truncate = UniformSymmetricTruncate.apply
        elif self.quant_mode == "asymmetric":
            raise NotImplementedError
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

    def forward(self, x):

        self.update_minmax(self.conv.weight)

        if self.use_scaling:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.conv.weight,
                        self.weight_alpha[str(self.precision_level)],
                        self.precision_level,
                        self.scale_factor,
                    )
        else:
            self.scale_factor = symmetric_linear_quantization_params(self.precision_level, self.min, self.max)
            self.quant_weight = self.weight_function(
                        self.conv.weight,
                        self.weight_alpha,
                        self.precision_level,
                        self.scale_factor,
                    )

        x = F.conv2d(
            x,
            self.quant_weight, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )

        return x

    def __repr__(self):
        s = super(QuantConv2d, self).__repr__()
        s = "(" + s + " weight_bit={}, quantize_mode={})".format(
            self.weight_bit, self.quant_mode)
        return s
