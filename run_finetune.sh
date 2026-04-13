#!/bin/bash
# Activate your Python environment (adjust path as needed)
# conda activate /path/to/your/env

export CUDA_VISIBLE_DEVICES="0"

python finetune.py \
    --num_precisions=1 \
    --base_model="/path/to/hubert-large-960h" \
    --quant_model="./exp/hubert_large/SepCB_rvq_nc4_nb8_gs8_bs1_fbs32_lbs32_ft10_ns2048_4bit" \
    --output_dir="./results/hubert_large/finetune_4bit" \
    --seed=1000 \
    --dataloader_num_workers=4 \
    --per_device_train_batch_size=4 \
    --gradient_accumulation_steps=2 \
    --per_device_eval_batch_size=1 \
    --num_train_epochs=1 \
    --learning_rate=2e-5 \
    --warmup_ratio=0.1 \
    --use_fp16
