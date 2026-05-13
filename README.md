# Extremely Low-Bit ASR

Code for *"Towards Extremely Low-Bit and Multi-Precision Conformer and Speech Foundation Model Quantization"* (IEEE/ACM Transactions on Audio, Speech, and Language Processing Submission).

This repository contains two methods:

## [CSVQ](./CSVQ/) — Codebook-Shared Vector Quantization

Post-training quantization for HuBERT speech foundation models, evaluated on LibriSpeech CTC.

- Multi-precision support (2/3/4-bit) with shared codebooks
- Based on HuggingFace HuBERT-Large-960h
- Calibration + optional finetuning pipeline

## [QACT](./QACT/) — Quantization-Aware Co-Training

Quantization-aware training for Conformer ASR on Switchboard, based on ESPnet v0.10.7a1.

- Joint 2-bit / 1-bit co-training with shared weights
- KL-divergence regularization from high-precision to low-precision sub-networks
- Stochastic precision scheduling
- Learnable per-tensor scaling factor

Please refer to each subdirectory's README for detailed setup and usage instructions.

