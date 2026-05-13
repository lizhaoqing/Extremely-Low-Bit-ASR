import argparse
from itertools import groupby
import logging
import math
import os

import chainer
from chainer import reporter
import numpy as np
import six
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.e2e_asr_common import label_smoothing_dist
from espnet.nets.pytorch_backend.ctc import ctc_for
from espnet.nets.pytorch_backend.frontends.feature_transform import (
    feature_transform_for,
)
from espnet.nets.pytorch_backend.frontends.frontend import frontend_for
from espnet.nets.pytorch_backend.initialization import lecun_normal_init_parameters
from espnet.nets.pytorch_backend.initialization import set_forget_bias_to_one
from espnet.nets.pytorch_backend.nets_utils import get_subsample
from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.nets_utils import to_torch_tensor
from espnet.nets.pytorch_backend.rnn.argument import (
    add_arguments_rnn_encoder_common,
    add_arguments_rnn_decoder_common,
    add_arguments_rnn_attention_common,
)
from espnet.nets.pytorch_backend.rnn.attentions import att_for
from espnet.nets.pytorch_backend.rnn.decoders import decoder_for
from espnet.nets.pytorch_backend.rnn.encoders import encoder_for
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.utils.fill_missing_args import fill_missing_args

CTC_LOSS_THRESHOLD = 10000


class Reporter(chainer.Chain):
    """A chainer reporter wrapper."""

    def report(self, precision, loss_ctc, loss_att, acc, cer_ctc, cer, wer, mtl_loss):
        """Report at every step."""
        reporter.report({"precision": precision}, self)
        reporter.report({"loss_ctc": loss_ctc}, self)
        reporter.report({"loss_att": loss_att}, self)
        reporter.report({"acc": acc}, self)
        reporter.report({"cer_ctc": cer_ctc}, self)
        reporter.report({"cer": cer}, self)
        reporter.report({"wer": wer}, self)
        logging.info("mtl loss:" + str(mtl_loss))
        reporter.report({"loss": mtl_loss}, self)
