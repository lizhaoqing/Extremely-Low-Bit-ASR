#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ConvolutionModule with quantization."""

from torch import nn
from model.quant_modules import *


class QuantConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model with quantized weights.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 activation=nn.ReLU(), bias=True,
                 shared_layer_num=1,
                 ):
        """Construct an ConvolutionModule object."""
        super(QuantConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=channels, bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = activation

        self.qpointwise_conv1 = QuantConv1d(
            self.pointwise_conv1, weight_bit,
            weight_alpha, bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )
        self.qdepthwise_conv = QuantConv1d(
            self.depthwise_conv, weight_bit,
            weight_alpha, bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )
        self.qpointwise_conv2 = QuantConv1d(
            self.pointwise_conv2, weight_bit,
            weight_alpha, bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.qpointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.qdepthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.qpointwise_conv2(x)

        return x.transpose(1, 2)


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model (full-precision).

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernerl size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=nn.ReLU(), bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels, 2 * channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels, channels, kernel_size, stride=1,
            padding=(kernel_size - 1) // 2, groups=channels, bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels, channels, kernel_size=1, stride=1, padding=0, bias=bias,
        )
        self.activation = activation

    def forward(self, x):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        x = x.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)
