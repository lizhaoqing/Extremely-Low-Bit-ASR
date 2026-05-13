#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quantized Transformer Decoder."""

import logging

from typing import Any
from typing import List
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from model.quant_attention import QuantMultiHeadedAttention
from model.quant_decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.dynamic_conv import DynamicConvolution
from espnet.nets.pytorch_backend.transformer.dynamic_conv2d import DynamicConvolution2D
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.lightconv import LightweightConvolution
from espnet.nets.pytorch_backend.transformer.lightconv2d import LightweightConvolution2D
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from model.quant_feedforward import (
    QuantPositionwiseFeedForward,
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface
from model.quant_modules import QuantModule, SwitchLayerNorm


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class Decoder(BatchScorerInterface, torch.nn.Module):
    """Transformer decoder module.

    Args:
        odim (int): Output diminsion.
        self_attention_layer_type (str): Self-attention layer type.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        self_attention_dropout_rate (float): Dropout rate in self-attention.
        src_attention_dropout_rate (float): Dropout rate in source-attention.

    """

    def __init__(
        self,
        odim,
        selfattention_layer_type="selfattn",
        attention_dim=256,
        attention_heads=4,
        conv_wshare=4,
        conv_kernel_length=11,
        conv_usebias=False,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        input_layer="embed",
        use_output_layer=True,
        pos_enc_class=PositionalEncoding,
        normalize_before=True,
        concat_after=False,
        weight_bit=8,
        weight_alpha=None,
        bias_bit=None,
        quant_mode='symmetric',
        per_channel=False,
        use_scaling=False,
        weight_percentile=0,
        shared_layer_num=1,
    ):
        """Construct an Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)
        if input_layer == "embed":
            self.embed = torch.nn.Sequential(
                torch.nn.Embedding(odim, attention_dim),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif input_layer == "linear":
            self.embed = torch.nn.Sequential(
                torch.nn.Linear(odim, attention_dim),
                torch.nn.LayerNorm(attention_dim),
                torch.nn.Dropout(dropout_rate),
                torch.nn.ReLU(),
                pos_enc_class(attention_dim, positional_dropout_rate),
            )
        elif isinstance(input_layer, torch.nn.Module):
            self.embed = torch.nn.Sequential(
                input_layer, pos_enc_class(attention_dim, positional_dropout_rate)
            )
        else:
            raise NotImplementedError("only `embed` or torch.nn.Module is supported.")
        self.normalize_before = normalize_before

        # self-attention module definition
        if selfattention_layer_type == "selfattn":
            logging.info("decoder self-attention layer type = self-attention")
            decoder_selfattn_layer = QuantMultiHeadedAttention
            decoder_selfattn_layer_args = [
                {
                    'n_head': attention_heads,
                    'n_feat': attention_dim,
                    'dropout_rate': self_attention_dropout_rate,
                    'weight_bit': weight_bit,
                    'weight_alpha': weight_alpha,
                    'bias_bit': bias_bit,
                    'quant_mode': quant_mode,
                    'per_channel': per_channel,
                    'use_scaling': use_scaling,
                    'weight_percentile': weight_percentile,
                    'shared_layer_num': shared_layer_num,
                }
            ] * num_blocks

        positionwise_layer = QuantPositionwiseFeedForward
        positionwise_layer_args = {
            'idim': attention_dim,
            'hidden_units': linear_units,
            'dropout_rate': dropout_rate,
            'weight_bit': weight_bit,
            'weight_alpha': weight_alpha,
            'bias_bit': bias_bit,
            'quant_mode': quant_mode,
            'per_channel': per_channel,
            'use_scaling': use_scaling,
            'weight_percentile': weight_percentile,
            'shared_layer_num': shared_layer_num,
        }

        decoder_srcattn_layer = QuantMultiHeadedAttention
        decoder_srcattn_layer_args = {
            'n_head': attention_heads,
            'n_feat': attention_dim,
            'dropout_rate': src_attention_dropout_rate,
            'weight_bit': weight_bit,
            'weight_alpha': weight_alpha,
            'bias_bit': bias_bit,
            'quant_mode': quant_mode,
            'per_channel': per_channel,
            'use_scaling': use_scaling,
            'weight_percentile': weight_percentile,
            'shared_layer_num': shared_layer_num,
        }

        self.decoders = repeat(
            num_blocks,
            lambda lnum: DecoderLayer(
                attention_dim,
                decoder_selfattn_layer(**decoder_selfattn_layer_args[lnum]),
                decoder_srcattn_layer(**decoder_srcattn_layer_args),
                positionwise_layer(**positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        self.selfattention_layer_type = selfattention_layer_type
        if self.normalize_before:
            self.after_norm = LayerNorm(attention_dim)
        if use_output_layer:
            self.output_layer = torch.nn.Linear(attention_dim, odim)
        else:
            self.output_layer = None

        self.num_blocks = num_blocks

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            memory_mask (torch.Tensor): Encoded memory mask (#batch, maxlen_in).

        Returns:
            torch.Tensor: Decoded token score before softmax (#batch, maxlen_out, odim).
            torch.Tensor: Score mask before softmax (#batch, maxlen_out).

        """
        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        if self.normalize_before:
            x = self.after_norm(x)
        if self.output_layer is not None:
            x = self.output_layer(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, cache=None):
        """Forward one step.

        Args:
            tgt (torch.Tensor): Input token ids, int64 (#batch, maxlen_out).
            tgt_mask (torch.Tensor): Input token mask (#batch, maxlen_out).
            memory (torch.Tensor): Encoded memory, float32 (#batch, maxlen_in, feat).
            cache (List[torch.Tensor]): List of cached tensors.

        Returns:
            torch.Tensor: Output tensor (batch, maxlen_out, odim).
            List[torch.Tensor]: List of cache tensors of each decoder layer.

        """
        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, None, cache=c
            )
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]
        if self.output_layer is not None:
            y = torch.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        if self.selfattention_layer_type != "selfattn":
            logging.warning(
                f"{self.selfattention_layer_type} does not support cached decoding."
            )
            state = None
        logp, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch.

        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.

        """
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            batch_state = [
                torch.stack([states[b][i] for b in range(n_batch)])
                for i in range(n_layers)
            ]

        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        logp, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def set_precision_level(self, precision):
        def _fix(m):
            for child in m.children():
                if isinstance(child, QuantModule):
                    child.set_precision(precision)
                else:
                    _fix(child)

        _fix(self)

    def set_layerwise_precision(self, prec_list):
        def _fix(m, prec):
            for child in m.children():
                if isinstance(child, QuantModule):
                    child.set_precision(prec)
                else:
                    _fix(child, prec)

        assert isinstance(prec_list, list)
        if len(prec_list) == 1:
            self.set_precision_level(prec_list[0])
        elif len(prec_list) == self.num_blocks:
            for i, enc_layer in enumerate(self.decoders):
                _fix(enc_layer, prec_list[i])
        else:
            raise ValueError('length of prec_list does not match the number of decoder layers')
