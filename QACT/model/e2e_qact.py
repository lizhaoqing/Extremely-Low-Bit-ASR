#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QACT: Quantization-Aware Co-Training for Conformer ASR.

This module implements the E2E model for QACT, which jointly trains
multiple precision sub-networks (e.g., 2-bit and 1-bit) with shared weights,
using KL-divergence regularization and stochastic precision scheduling.

Reference:
    "Towards Extremely Low-Bit and Multi-Precision Conformer and Speech
     Foundation Model Quantization"

It is a fusion of `e2e_asr_transformer.py`
Refer to: https://arxiv.org/abs/2005.08100

"""

from model.quant_encoder import Encoder
from model.quant_decoder import Decoder as QDecoder
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,
    verify_rel_pos_type,
)
from model.argument import add_arguments_bitsharing_common

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
import math
import torch
import logging
from argparse import Namespace
import numpy
from espnet.nets.ctc_prefix_score import CTCPrefixScore
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
from model.reporter import Reporter
from model.losses import LabelSoftLoss


class E2E(E2ETransformer):
    """E2E module for QACT.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_conformer_arguments(parser)
        E2E.add_bitsharing_arguments(parser)
        return parser

    @staticmethod
    def add_conformer_arguments(parser):
        """Add arguments for conformer model."""
        group = parser.add_argument_group("conformer model specific setting")
        group = add_arguments_conformer_common(group)
        return parser

    @staticmethod
    def add_bitsharing_arguments(parser):
        """Add arguments for QACT quantization."""
        group = parser.add_argument_group("QACT quantization specific setting")
        group = add_arguments_bitsharing_common(group)
        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        # Check the relative positional encoding type
        args = verify_rel_pos_type(args)

        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            zero_triu=args.zero_triu,
            cnn_module_kernel=args.cnn_module_kernel,
            stochastic_depth_rate=args.stochastic_depth_rate,
            intermediate_layers=self.intermediate_ctc_layers,
            ctc_softmax=self.ctc.softmax if args.self_conditioning else None,
            conditioning_layer_dim=odim,
            weight_bit=args.enc_weight_bit,
            weight_alpha=None,
            quant_mode=args.quant_mode,
            per_channel=args.per_channel,
            use_scaling=args.use_scaling,
            shared_layer_num=args.enc_shared_layer_num,
            quant_cnn=args.quant_cnn
        )

        if args.mtlalpha < 1:
            if not args.quant_decoder:
                self.decoder = Decoder(
                    odim=odim,
                    selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    conv_wshare=args.wshare,
                    conv_kernel_length=args.ldconv_decoder_kernel_length,
                    conv_usebias=args.ldconv_usebias,
                    linear_units=args.dunits,
                    num_blocks=args.dlayers,
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                )
            else:
                self.decoder = QDecoder(
                    odim=odim,
                    selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
                    attention_dim=args.adim,
                    attention_heads=args.aheads,
                    conv_wshare=args.wshare,
                    conv_kernel_length=args.ldconv_decoder_kernel_length,
                    conv_usebias=args.ldconv_usebias,
                    linear_units=args.dunits,
                    num_blocks=args.dlayers,
                    dropout_rate=args.dropout_rate,
                    positional_dropout_rate=args.dropout_rate,
                    self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                    weight_bit=args.dec_weight_bit,
                    per_channel=args.per_channel,
                    use_scaling=args.use_scaling,
                    shared_layer_num=args.dec_shared_layer_num,
                )

            self.soft_criterion = LabelSoftLoss(
                odim,
                ignore_id,
                args.transformer_length_normalized_loss,
            )
        else:
            self.decoder = None
            self.criterion = None

        self.num_blocks = args.elayers
        self.precision_levels = [[2], [1]]
        self.reporter = Reporter()
        self.soft_target_hs = None
        self.soft_target_pred = None

        self.lambda_1 = args.lambda_1
        self.lambda_2 = args.lambda_2

        # Stochastic Precision schedule
        if args.mix_rate == 1.:
            self.mix_rate = numpy.linspace(0.5, 0.8, 12)
        elif args.mix_rate == 0.:
            self.mix_rate = numpy.linspace(0.8, 0.2, 12)
        elif args.mix_rate == 0.8:
            a = numpy.linspace(numpy.log(0.2), numpy.log(0.8), 12)
            self.mix_rate = numpy.exp(a / 0.9)
        elif args.mix_rate == 0.2:
            a = numpy.linspace(numpy.log(0.8), numpy.log(0.2), 12)
            self.mix_rate = numpy.exp(a / 0.9)
        elif args.mix_rate == 1.8:
            # Log-linear schedule (used in paper)
            a = numpy.linspace(numpy.exp(0.2), numpy.exp(0.8), 12)
            self.mix_rate = numpy.log(a / 0.9)
        else:
            raise ValueError('the mix_rate should be set to 0, 0.2, 0.8, 1, or 1.8')

        self.reset_parameters(args)

    def forward(self, xs_pad, ilens, ys_pad):
        """Forward pass with co-training.

        Runs forward for each precision level (2-bit, 1-bit, and a random
        stochastic-precision configuration), averaging the losses.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: loss value
        :rtype: torch.Tensor
        """
        assert len(self.precision_levels) > 0
        loss = []
        acc = []
        prec_list = []
        for i in range(self.num_blocks):
            prec_list = prec_list + [2] if numpy.random.rand() > self.mix_rate[i] else prec_list + [1]
        prec_list = self.precision_levels + [prec_list]

        self.soft_target_hs = None
        self.soft_target_pred = None
        self.report = True
        for precision in prec_list:
            self.encoder.set_layerwise_precision(precision)
            loss.append(self.subforward(xs_pad, ilens, ys_pad, precision[0]))
            acc.append(self.acc)

        if not self.training:
            logging.warning(str(self.precision_levels) + ', mixed: ' + str(acc))

        self.loss = sum(loss) / len(loss)
        return self.loss

    def subforward(self, xs_pad, ilens, ys_pad, precision):
        """E2E forward for a single precision configuration.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :param int precision: current precision level
        :return: loss value
        :rtype: torch.Tensor
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask, hs_intermediates = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # 2. forward decoder
        if self.decoder is not None:
            ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
            )
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss with KD regularization
            if self.soft_target_pred is None:
                # First pass (2-bit): use hard labels only, store soft targets
                loss_att = self.criterion(pred_pad, ys_out_pad)
                self.soft_target_pred = torch.softmax(pred_pad, dim=-1).detach()
            else:
                # Subsequent passes (1-bit, mixed): use KD loss + CE loss
                loss_att = self.soft_criterion(pred_pad, self.soft_target_pred, ys_out_pad) * self.lambda_2 + self.criterion(pred_pad, ys_out_pad) * self.lambda_1
            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )
        else:
            loss_att = None
            self.acc = None

        cer_ctc = None
        loss_intermediate_ctc = 0.0
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            if self.soft_target_hs is None:
                # First pass (2-bit): use hard CTC loss, store soft targets
                loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
                self.soft_target_hs = self.ctc.softmax(hs_pad).detach()
            else:
                # Subsequent passes: use KD CTC loss + hard CTC loss
                loss_ctc = self.soft_criterion(self.ctc.ctc_lo(hs_pad), self.soft_target_hs, torch.ones((hs_pad.shape[0], hs_pad.shape[1])).to(hs_pad.device)) * self.lambda_2 + \
                            self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad) * self.lambda_1
            if not self.training and self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

            if self.intermediate_ctc_weight > 0 and self.intermediate_ctc_layers:
                for hs_intermediate in hs_intermediates:
                    loss_inter = self.ctc(
                        hs_intermediate.view(batch_size, -1, self.adim), hs_len, ys_pad
                    )
                    loss_intermediate_ctc += loss_inter

                loss_intermediate_ctc /= len(self.intermediate_ctc_layers)

        # 5. compute cer/wer
        if self.training or self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # compute final loss
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    1 - self.intermediate_ctc_weight
                ) * loss_ctc + self.intermediate_ctc_weight * loss_intermediate_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            if self.intermediate_ctc_weight > 0:
                self.loss = (
                    (1 - alpha - self.intermediate_ctc_weight) * loss_att
                    + alpha * loss_ctc
                    + self.intermediate_ctc_weight * loss_intermediate_ctc
                )
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if self.report:
            self.report = False
            if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
                self.reporter.report(
                    precision, loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
                )
            else:
                logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    def encode(self, x, precision):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :param int precision: precision level for encoding
            1 = all 1-bit
            2 = all 2-bit
            11 = first 6 layers 1-bit, last 6 layers 2-bit (1.5-bit avg)
            12 = first 6 layers 2-bit, last 6 layers 1-bit (1.5-bit avg)
            13 = layers 1-3 1-bit, 4-9 2-bit, 10-12 1-bit
            14 = layers 1-3 2-bit, 4-9 1-bit, 10-12 2-bit
            15 = alternating 1-bit/2-bit
            0 = random 6 layers at 1-bit, 6 layers at 2-bit
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        if precision == 11:
            precision = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        elif precision == 12:
            precision = [2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1]
        elif precision == 13:
            precision = [1, 1, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1]
        elif precision == 14:
            precision = [2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 2]
        elif precision == 15:
            precision = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
        elif precision == 0:
            idx = numpy.random.choice(12, 6, 0)
            precision = torch.tensor([2] * 12)
            precision[idx] = 1
            precision = precision.tolist()
            logging.warning(f"precision is sampled as {precision}")
        else:
            precision = [precision]
        self.encoder.set_layerwise_precision(precision)
        enc_output, _, _ = self.encoder(x, None)
        return enc_output.squeeze(0)
