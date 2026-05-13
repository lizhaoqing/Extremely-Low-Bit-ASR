#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Multi-Head Attention layer with quantization."""

import math

import numpy
import torch
from torch import nn
from model.quant_modules import *


class QuantMultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, n_head, n_feat, dropout_rate,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 shared_layer_num=1,
                ):
        """Construct an MultiHeadedAttention object."""
        super(QuantMultiHeadedAttention, self).__init__()
        self.n_feat = n_feat
        self.d_k = n_feat // 4
        self.h = n_head
        self.adim = self.d_k * n_head
        self.linear_q = nn.Linear(n_feat, self.adim)
        self.linear_k = nn.Linear(n_feat, self.adim)
        self.linear_v = nn.Linear(n_feat, self.adim)
        self.linear_out = nn.Linear(self.adim, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

        self.qlinear_q = QuantLinear(
            self.linear_q, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )
        self.qlinear_k = QuantLinear(
            self.linear_k, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )
        self.qlinear_v = QuantLinear(
            self.linear_v, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )
        self.qlinear_out = QuantLinear(
            self.linear_out, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )

    def qforward_qkv(self, query, key, value):

        n_batch = query.size(0)
        q = self.qlinear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.qlinear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.qlinear_v(value).view(n_batch, -1, self.h, self.d_k)
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)
        return q, k, v

    def qforward_attention(self, value, scores, mask):

        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = float(
                numpy.finfo(torch.tensor(
                    0, dtype=scores.dtype).numpy().dtype).min
            )
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn) / self.adim * self.n_feat

        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)
        return self.qlinear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):

        q, k, v = self.qforward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / \
            math.sqrt(self.d_k)
        return self.qforward_attention(v, scores, mask)


class QuantLegacyRelPositionMultiHeadedAttention(QuantMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (old version).

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 zero_triu=False,
                 shared_layer_num=1,
                 ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head, n_feat, dropout_rate, weight_bit,
            weight_alpha, bias_bit, quant_mode,
            per_channel, use_scaling,
            weight_percentile, shared_layer_num,
        )

        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, self.adim, bias=False)
        self.qlinear_pos = QuantLinear(
            self.linear_pos, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )

        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, head, time1, time2).

        Returns:
            torch.Tensor: Output tensor.

        """
        zero_pad = torch.zeros(
            (*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):

        q, k, v = self.qforward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.qlinear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        # (batch, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # (batch, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch, head, time1, time1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(
            self.d_k
        )  # (batch, head, time1, time2)
        return self.qforward_attention(v, scores, mask)


class QuantRelPositionMultiHeadedAttention(QuantMultiHeadedAttention):
    """Multi-Head Attention layer with relative position encoding (new implementation).

    Paper: https://arxiv.org/abs/1901.02860

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.

    """

    def __init__(self, n_head, n_feat, dropout_rate,
                 weight_bit=8,
                 weight_alpha=None,
                 bias_bit=None,
                 quant_mode='symmetric',
                 per_channel=False,
                 use_scaling=False,
                 weight_percentile=0,
                 zero_triu=False,
                 shared_layer_num=1,
                 ):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(
            n_head, n_feat, dropout_rate, weight_bit,
            weight_alpha, bias_bit, quant_mode,
            per_channel, use_scaling,
            weight_percentile, shared_layer_num,
        )

        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, self.adim, bias=False)
        self.qlinear_pos = QuantLinear(
            self.linear_pos, weight_bit, weight_alpha,
            bias_bit, quant_mode, per_channel,
            use_scaling, weight_percentile,
            shared_layer_num,
        )

        # these two learnable bias are used in matrix c and matrix d
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """Compute relative positional encoding."""
        zero_pad = torch.zeros(
            (*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        x = x_padded[:, :, 1:].view_as(x)

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)))
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):

        q, k, v = self.qforward_qkv(query, key, value)
        q = q.transpose(1, 2)  # (batch, time1, head, d_k)

        n_batch_pos = pos_emb.size(0)
        p = self.qlinear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        p = p.transpose(1, 2)  # (batch, head, time1, d_k)

        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)
        return self.qforward_attention(v, scores, mask)
