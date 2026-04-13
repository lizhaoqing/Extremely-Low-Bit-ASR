#!/bin/bash
# Activate your Python environment (adjust path as needed)
# conda activate /path/to/your/env
# or: source /path/to/venv/bin/activate

# Set data paths (required)
export LIBRISPEECH_DATA_DIR=/path/to/librispeech/   # directory containing LibriSpeech/train-clean-100/ etc.
export LIBRISPEECH_CACHE_DIR=./data_cache/

export CUDA_VISIBLE_DEVICES="0"
export MODEL_PATH=/path/to/hubert-large-960h         # HuggingFace model dir or "facebook/hubert-large-ls960-ft"
export DATASET_PATH=librispeech

# Optional: wandb logging
# export WANDB_PROJECT=CSVQ
# export WANDB_NAME=SepCB_rvq_nc2_1_1_nb8_gs8_p234

export SAVE_PATH=./exp/hubert_large/SepCB_rvq_nc2_1_1_nb8_gs8_p234

# ========== Multi-precision 2/3/4-bit quantization ==========
python quantize.py $MODEL_PATH $DATASET_PATH \
 --nsamples=2048 \
 --val_size=128 \
 --num_codebooks="2,1,1" \
 --nbits_per_codebook="8,8,8" \
 --in_group_size=8 \
 --relative_mse_tolerance=0.01 \
 --finetune_batch_size=32 \
 --finetune_max_epochs=10 \
 --finetune_early_stop=3 \
 --finetune_keep_best \
 --local_batch_size=32 \
 --offload_activations \
 --resume \
 --num_precisions=3 \
 --beam_size=1 \
 --save $SAVE_PATH
 # --wandb   # uncomment to enable wandb logging
