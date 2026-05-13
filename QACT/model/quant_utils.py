import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Function, Variable


def clamp(input, min, max, inplace=False):
    """
    Clamp tensor input to (min, max).
    input: input tensor to be clamped
    """
    if inplace:
        input.clamp_(min, max)
        return input
    return torch.clamp(input, min, max)


def get_percentile_min_max(input, lower_percentile, upper_percentile, output_tensor=False):
    """
    Calculate the percentile max and min values in a given tensor
    Parameters:
    ----------
    input: tensor
        the tensor to calculate percentile max and min
    lower_percentile: float
        if 0.1, means we return the value of the smallest 0.1% value in the tensor as percentile min
    upper_percentile: float
        if 99.9, means we return the value of the largest 0.1% value in the tensor as percentile max
    output_tensor: bool, default False
        if True, this function returns tensors, otherwise it returns values
    """
    input_length = input.shape[0]

    lower_index = round(input_length * (1 - lower_percentile * 0.01))
    upper_index = round(input_length * upper_percentile * 0.01)

    upper_bound = torch.kthvalue(input, k=upper_index).values

    if lower_percentile == 0:
        lower_bound = upper_bound * 0
    else:
        lower_bound = -torch.kthvalue(-input, k=lower_index).values

    if not output_tensor:
        lower_bound = lower_bound.item()
        upper_bound = upper_bound.item()
    return lower_bound, upper_bound


def symmetric_linear_quantization_params(
    num_bits,
    saturation_min,
    saturation_max,
    per_channel=False
):
    """
    Compute the scaling factor with the given quantization range for symmetric quantization.
    Parameters:
    ----------
    saturation_min: lower bound for quantization range
    saturation_max: upper bound for quantization range
    per_channel: if True, calculate the scaling factor per channel.
    """

    with torch.no_grad():

        if num_bits == 1:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8)
            return scale

        n = 2 ** (num_bits - 1) - 1
        if per_channel:
            scale, _ = torch.max(torch.stack([saturation_min.abs(), saturation_max.abs()], dim=1), dim=1)
            scale = torch.clamp(scale, min=1e-8) / n
        else:
            scale = max(saturation_min.abs(), saturation_max.abs())
            scale = torch.clamp(scale, min=1e-8) / n

    return scale


def asymmetric_linear_quantization_params(
    num_bits,
    saturation_min,
    saturation_max,
    integral_zero_point=True
):
    """
    Compute the scaling factor and zeropoint with the given quantization range for asymmetric quantization.
    """

    with torch.no_grad():
        n = 2 ** num_bits - 1
        scale = torch.clamp((saturation_max - saturation_min), min=1e-8) / float(n)

        zero_point = -saturation_min / scale

        if integral_zero_point:
            if isinstance(zero_point, torch.Tensor):
                zero_point = zero_point.round()
            else:
                zero_point = float(round(zero_point))

        return scale, zero_point


def uniform_quantization(tensor, alpha, bit, scale_factor):

    if bit == 1:
        data = tensor / scale_factor / alpha
        data_q = data.sign()
        data_q = data_q * alpha * scale_factor
        return data_q

    n = 2 ** (bit - 1) - 1
    data = tensor / scale_factor / alpha
    data = data.clamp(-n, n)
    data_q = (data.round() - data).detach() + data
    data_q = data_q * alpha * scale_factor
    return data_q


def uniform_truncate(tensor, alpha, bit, hbit, scale_factor):
    data_q = uniform_quantization(tensor, alpha, bit, scale_factor)
    data_fp = data_q / scale_factor / alpha
    data_hp = (data_fp / (2**(bit-hbit) * 1.0)).floor()
    data_hp *= (2**(bit-hbit) * 1.0)
    data_hp = data_hp * alpha * scale_factor
    return data_hp


class ScaleSymmetricLinearQuant_old(Function):
    @staticmethod
    def forward(ctx, input, alpha, bit, scale_factor):

        if bit == 1:
            data = input / scale_factor / alpha
            data_q = data.sign()
            ctx.save_for_backward(data, data_q)
            data_q = data_q * alpha * scale_factor
            return data_q

        n = 2 ** (bit - 1) - 1

        input_s = input / scale_factor / alpha
        input_n = input_s.clamp(-n, n)
        input_q = input_n.round()
        ctx.save_for_backward(input_s / n, input_q / n)
        input_q = input_q.mul(scale_factor * alpha)

        return input_q

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        input_s, input_temp = ctx.saved_tensors
        i = (input_s.abs()>1.).float()
        sign = input_s.sign()
        grad_alpha = (grad_output * (sign * i + (input_temp - input_s) * (1 - i))).sum()

        return grad_input, grad_alpha, None, None


class UniformSymmetricLinearQuant(Function):
    @staticmethod
    def forward(ctx, input, alpha, bit, scale_factor):

        return uniform_quantization(input, alpha, bit, scale_factor)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None

class UniformSymmetricTruncate(Function):
    @staticmethod
    def forward(ctx, input, alpha, bit, hbit, scale_factor):

        return uniform_truncate(input, alpha, bit, hbit, scale_factor)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None, None


class InplaceSymmetricLinearQuant(Function):
    @staticmethod
    def forward(ctx, input, alpha, bit, scale_factor):
        n = 2 ** (bit - 1) - 1
        input.mul_(1. / scale_factor / alpha).clamp_(-n, n).round_()
        input.mul_(scale_factor * alpha)
        return input

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output, None, None, None
