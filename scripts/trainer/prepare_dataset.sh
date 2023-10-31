#!/bin/bash
#set -e

#export MODEL_PATH="t5-small"
#export MODEL_TYPE="t5"

#export MODEL_PATH="mistralai/Mixtral-8x7B-Instruct-v0.1"
#export MODEL_TYPE="mixtral"

#export MODEL_TYPE="bloom"
#export MODEL_PATH="bigscience/bloom-560m"

export MODEL_TYPE="llama"
export MODEL_PATH="meta-llama/Llama-2-7b-hf"

export CUSTOMIZED_TOKENIZER="output/customed_tokenizer/llama"
export OUPUT_DIR="dataset"

# export LLM_ARCHITECT="encoder-decoder"
export LLM_ARCHITECT="decoder-only"

python nlp_pipeline/dataset/build_dataset.py \
    --output_dir $OUPUT_DIR \
    --model_type $MODEL_TYPE \
    --llm_architect $LLM_ARCHITECT \
    --model_name_or_path $MODEL_PATH \
    --pretrained_tokenizer_name_or_path $MODEL_PATH \
    --customized_tokenizer_save_path $CUSTOMIZED_TOKENIZER \
    --dataset_train_path data/requirement_testcase.json \
    --dataset_valid_path data/requirement_testcase.json \
    --dataset_test_path data/requirement_testcase.json \
    --max_source_length 128 \
    --max_target_length 512 \
    --train_file_name train_data_hl_t5.pt \
    --valid_file_name valid_data_hl_t5.pt \
    --test_file_name test_data_hl_t5.pt
