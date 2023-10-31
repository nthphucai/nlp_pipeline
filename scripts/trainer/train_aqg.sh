#!/bin/bash
set -e

export MODEL_PATH="t5-small"
export OUTPUT_DIR="output/models"
export TOKENIZER_PATH="t5_qg_tokenizer"

python nlp_pipeline/trainer/run_aqg.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --model_type t5 \
    --project_name question_generation_multitask \
    --output_dir $OUTPUT_DIR \
    --train_file_path output/data/question-generation/train_data_hl_t5.pt \
    --valid_file_path output/data/question-generation/valid_data_hl_t5.pt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-5 \
    --learning_rate 1e-5 \
    --weight_decay 1e-6 \
    --num_train_epochs 10 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --seed 42 \
    --do_train \
    --do_eval \
    --remove_unused_columns False \
    --logging_steps 2 \
    --report_to wandb \
    --evaluation_strategy epoch \
    --logging_first_step True \
    --save_strategy epoch \
    --save_total_limit 1 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --metric_for_best_model eval_loss \
    --auto_find_batch_size True \
    --overwrite_output_dir True
    