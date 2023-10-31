#!/bin/bash
set -e

export OUTPUT_PATH="data/simple-question/simple_question_gcn-t5-base-multitask-hl.txt"
export REFERENCE_PATH="data/test_references.txt"
export EVAL_PATH="output/data/history/simple/question-answering/valid_data_hl_t5.pt"
export MODEL_PATH="output/models/history/gcn/v1.2"

python questgen/trainer/eval_aqg.py \
    --task multitask \
    --model_name_or_path $MODEL_PATH \
    --valid_file_path $EVAL_PATH \
    --tokenizer_name_or_path  $MODEL_PATH \
    --model_type t5 \
    --num_beams 1 \
    --max_decoding_length 64 \
    --repetition_penalty 1.5 \
    --length_penalty 1.0 \
    --reference_path $REFERENCE_PATH \
    --output_path $OUTPUT_PATH \
    --device gpu

nlg-eval --hypothesis $OUTPUT_PATH \
        --references $REFERENCE_PATH \
        --no-skipthoughts \
        --no-glove
