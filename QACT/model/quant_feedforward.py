#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Positionwise feed forward layer with quantization."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.quant_modules import *


class QuantPositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 activation=torch.nn.ReLU(),
                 shared_layer_num=1,
                 ):
        """Construct an PositionwiseFeedForward object."""
        super(QuantPositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

        self.qw_1 = QuantLinear(self.w_1, weight_bit, weight_alpha,
                                     bias_bit, quant_mode, per_channel,
                                     use_scaling, weight_percentile,
                                     shared_layer_num,
                                     )
        self.qw_2 = QuantLinear(self.w_2, weight_bit, weight_alpha,
                                     bias_bit, quant_mode, per_channel,
                                     use_scaling, weight_percentile,
                                     shared_layer_num,
                                     )

    def forward(self, x):
        """Forward function."""
        return self.qw_2(self.dropout(self.activation(self.qw_1(x))))
