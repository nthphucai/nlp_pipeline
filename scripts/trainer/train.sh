#!/bin/bash
#set -e

#export MODEL_PATH="t5-small"
#export MODEL_TYPE="t5"

#export MODEL_PATH="mistralai/Mixtral-7B-v0.1"
#export MODEL_TYPE="mixtral"

export MODEL_PATH="meta-llama/Llama-2-7b-hf"
export MODEL_TYPE="llama"

#export MODEL_PATH="bigscience/bloom-560m"
#export MODEL_TYPE="bloom"

export OUTPUT_DIR="output/models/adapter_llama2"
export MODEL_ARCHITECT="decoder-only"

export CUSTOMIZED_TOKENIZER="output/customed_tokenizer/llama"

python nlp_pipeline/trainer/train.py \
    --model_name_or_path $MODEL_PATH \
    --tokenizer_name_or_path $CUSTOMIZED_TOKENIZER \
    --cache_dir $OUTPUT_DIR \
    --model_type $MODEL_TYPE \
    --llm_architect $MODEL_ARCHITECT \
    --project_name test_case_generation \
    --output_dir $OUTPUT_DIR \
    --train_file_path dataset/train_data_hl_t5.pt \
    --valid_file_path dataset/valid_data_hl_t5.pt \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --weight_decay 1e-6 \
    --optim paged_adamw_32bit \
    --save_steps 10 \
    --logging_steps 10 \
    --eval_steps 10 \
    --eval_accumulation_steps 10 \
    --learning_rate 5e-4 \
    --max_grad_norm 0.3 \
    --max_steps 30 \
    --warmup_ratio 0.03 \
    --evaluation_strategy steps \
    --lr_scheduler_type constant \
    --remove_unused_columns False \
    --overwrite_output_dir True \
    --seed 42 \
    --do_train True \
    --do_eval False \
    --report_to none \
    --logging_first_step False \
    --save_total_limit 1 \
    --load_best_model_at_end False \
    --greater_is_better False \
    --fp16 False \
    # --auto_find_batch_size True \
    # --metric_for_best_model eval_loss \
    # --group_by_length True \
    # --num_train_epochs 1 \
    # --eval_strategy epoch \
    # --save_strategy epoch \
