#!/bin/bash


accelerate launch finetune.py \
    --model_name_or_path output_qwen14b_checkpoints_new/checkpoint-1900 \
    --data_path ../datasets/agent_train_data.json \
    --bf16 False \
    --output_dir output_qwen14b_checkpoints_new \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --neftune_noise_alpha 5 \
    --save_total_limit 1 \
    --learning_rate 1e-6 \
    --weight_decay 0.01 \
    --adam_beta2 0.95 \
    --lr_scheduler_type "cosine" \
    --warmup_ratio 0.01 \
    --report_to "tensorboard" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --lazy_preprocess True \
    --use_lora False \
    --q_lora False \
    --deepspeed ds_config_zero3_new.json \
    --gradient_checkpointing