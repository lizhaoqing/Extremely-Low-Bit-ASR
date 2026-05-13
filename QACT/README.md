# QACT: Quantization-Aware Co-Training for Conformer ASR

This repository contains the implementation of QACT (Quantization-Aware Co-Training) for extremely low-bit Conformer-based automatic speech recognition (ASR).

Reference: *"Towards Extremely Low-Bit and Multi-Precision Conformer and Speech Foundation Model Quantization"*, IEEE/ACM Transactions on Audio, Speech, and Language Processing, 2026.

## Algorithm Overview

QACT jointly trains multiple precision sub-networks (2-bit and 1-bit) that share the same set of weights, enabling a single model to support flexible inference at different bit-widths.

Key components:
1. **Multi-Precision Co-Training**: A single model is trained with shared weights across 2-bit and 1-bit precision levels simultaneously.
2. **KL-Divergence Regularization**: The high-precision (2-bit) sub-network serves as a teacher; its soft output targets guide the low-precision (1-bit) sub-network via KL-divergence loss.
3. **Stochastic Precision (SP)**: During training, each encoder layer is randomly assigned to 1-bit or 2-bit with a log-linear probability schedule, improving robustness to arbitrary layer-wise precision combinations.
4. **Learnable Scaling Factor (alpha)**: Each quantized layer has a learnable tensor-wise scaling factor, adding only ~204 extra parameters to the entire model (~44.6M parameters total).

Loss function for non-teacher passes:
```
L = lambda_1 * L_CE(hard_target) + lambda_2 * L_KL(soft_target_from_2bit)
```
where `lambda_1=0.5`, `lambda_2=1.0` by default.

## Prerequisites

### ESPnet Version
This code is based on **ESPnet v0.10.7a1** (the ESPnet1 branch, not ESPnet2).

### Installation

1. Install ESPnet following the official instructions:
   ```bash
   git clone https://github.com/espnet/espnet.git
   cd espnet
   git checkout v.0.10.7   # or the closest tag to v0.10.7a1
   cd tools && make
   ```

2. Install Kaldi (required for feature extraction and scoring):
   ```bash
   # Follow https://github.com/kaldi-asr/kaldi
   ```

3. Ensure the following Python packages are available:
   - PyTorch >= 1.8
   - chainer
   - numpy
   - configargparse
   - sentencepiece

### Data

This recipe uses the **Switchboard-1 (LDC97S62)** corpus for training and **eval2000 (Hub5'00)**, **RT-02**, **RT-03** for evaluation.

Follow the standard ESPnet Switchboard recipe (e.g., `egs/swbd/asr1`) to prepare data through stages 0-3:
- Stage 0: Data preparation
- Stage 1: Feature extraction (80-dim filterbank + pitch)
- Stage 2: Dictionary and BPE (2000 subwords) preparation
- Stage 3: JSON data file creation

## Setup

1. Copy this entire `QACT/` directory into an ESPnet Switchboard experiment directory:
   ```bash
   # Assuming your ESPnet SWBD recipe is at:
   # /path/to/espnet/egs/swbd/asr1/

   cp -r QACT/* /path/to/espnet/egs/swbd/asr1/
   ```

2. Ensure `path.sh` correctly points to your ESPnet root:
   ```bash
   # Edit path.sh to set:
   MAIN_ROOT=/path/to/espnet
   KALDI_ROOT=/path/to/espnet/tools/kaldi
   ```

3. Ensure `PYTHONPATH` includes the experiment directory (so that `model.*` modules can be imported):
   ```bash
   export PYTHONPATH=/path/to/espnet/egs/swbd/asr1:$PYTHONPATH
   ```

## Training

After data preparation (stages 0-3), run QACT training:

```bash
./run.sh --stage 4 --stop-stage 4 --train-config conf/train_qact.yaml
```

Key hyperparameters in `conf/train_qact.yaml`:
| Parameter | Value | Description |
|-----------|-------|-------------|
| `enc-weight-bit` | 2 | Maximum encoder weight bit-width |
| `use-scaling` | true | Enable learnable scaling factor alpha |
| `quant-cnn` | true | Quantize convolution module |
| `mix-rate` | 1.8 | Stochastic precision schedule (1.8 = log-linear) |
| `lambda-1` | 0.5 | Weight for hard-label CE loss |
| `lambda-2` | 1.0 | Weight for soft-label KD loss |
| `epochs` | 100 | Total training epochs |
| `batch-size` | 64 | Mini-batch size |
| `accum-grad` | 8 | Gradient accumulation steps |

Training takes approximately 5 days on a single GPU.

## Decoding

Decode at different precision levels:

```bash
# 2-bit (all layers at 2-bit)
./run.sh --stage 5 --stop-stage 5 --decode-config conf/decode_2bit.yaml

# 1-bit (all layers at 1-bit)
./run.sh --stage 5 --stop-stage 5 --decode-config conf/decode_1bit.yaml

# 1.5-bit mixed precision (first 6 layers 1-bit, last 6 layers 2-bit)
./run.sh --stage 5 --stop-stage 5 --decode-config conf/decode_1.5bit.yaml
```

The precision is controlled by the `decode-precision` field in the decode config:
| Value | Meaning |
|-------|---------|
| `1` | All 12 layers at 1-bit |
| `2` | All 12 layers at 2-bit |
| `11` | Layers 1-6 at 1-bit, layers 7-12 at 2-bit |
| `12` | Layers 1-6 at 2-bit, layers 7-12 at 1-bit |
| `13` | Layers 1-3 at 1-bit, 4-9 at 2-bit, 10-12 at 1-bit |
| `14` | Layers 1-3 at 2-bit, 4-9 at 1-bit, 10-12 at 2-bit |
| `15` | Alternating 1-bit / 2-bit |
| `0` | Random: 6 layers at 1-bit, 6 layers at 2-bit |

## Expected Results (Switchboard eval2000)

| Precision | SWBD WER (%) | CH WER (%) | Avg WER (%) |
|-----------|-------------|------------|-------------|
| 2-bit | 10.4 | 15.3 | 12.86 |
| 1.5-bit | ~11 | ~16 | ~13.5 |
| 1-bit | ~13 | ~19 | ~16 |

## File Structure

```
QACT/
├── README.md                     # This file
├── run.sh                        # Main training & decoding script
├── average.py                    # Model averaging script
├── conf/
│   ├── train_qact.yaml           # QACT training config
│   ├── specaug.yaml              # SpecAugment config
│   ├── decode_2bit.yaml          # Decode config for 2-bit
│   ├── decode_1bit.yaml          # Decode config for 1-bit
│   └── decode_1.5bit.yaml        # Decode config for 1.5-bit mixed
├── model/
│   ├── __init__.py
│   ├── e2e_qact.py               # Main E2E QACT model
│   ├── quant_encoder.py          # Quantized Conformer encoder
│   ├── quant_encoder_layer.py    # Quantized encoder layer
│   ├── quant_decoder.py          # Quantized Transformer decoder
│   ├── quant_decoder_layer.py    # Quantized decoder layer
│   ├── quant_attention.py        # Quantized multi-head attention
│   ├── quant_feedforward.py      # Quantized feed-forward network
│   ├── quant_convolution.py      # Quantized convolution module
│   ├── quant_modules.py          # QuantLinear, QuantConv1d, SwitchLayerNorm
│   ├── quant_utils.py            # Quantization functions (STE, scaling)
│   ├── losses.py                 # LabelSoftLoss for KD
│   ├── reporter.py               # Training reporter
│   └── argument.py               # QACT-specific argument definitions
└── recog/
    └── asr_recog.py              # Recognition / decoding script
```

## License

Apache 2.0
