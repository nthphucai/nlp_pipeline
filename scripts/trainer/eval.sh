#!/bin/bash
set -e

export OUTPUT_PATH="output/requirement-testcases-t5-large.txt"
export REFERENCE_PATH="data/test_references.txt"
export EVAL_PATH="output/dataset/valid_data_hl_t5.pt"
export MODEL_PATH="output/models/t5-model"
export TOKENIZER_PATH="output/models/t5-model"

python test_automation/trainer/eval.py \
    --model_name_or_path $MODEL_PATH \
    --valid_file_path $EVAL_PATH \
    --tokenizer_name_or_path $TOKENIZER_PATH \
    --model_type t5 \
    --num_beams 1 \
    --max_decoding_length 1024 \
    --repetition_penalty 1.0 \
    --length_penalty 1.0 \
    --reference_path $REFERENCE_PATH \
    --output_path $OUTPUT_PATH \
    --device gpu

nlg-eval --hypothesis $OUTPUT_PATH \
        --references $REFERENCE_PATH \
        --no-skipthoughts \
        --no-glove