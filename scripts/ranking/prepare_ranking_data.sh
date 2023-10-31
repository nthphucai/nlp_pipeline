#!/bin/bash
set -e

python questgen/ranking/prepare_rank_data.py \
    --model_name_or_path output/models/v1.2/t5-en-vi-base-multitask \
    --tokenizer_name_or_path output/models/v1.2/t5-en-vi-base-multitask \
    --train_batch_size 4 \
    --eval_batch_size 4 \
    --test_batch_size 1 \
    --train_data_path data/qa_ranking/json/v1.3/train_data_v1.3.json \
    --eval_data_path data/qa_ranking/json/v1.3/dev_data_v1.3.json \
    --max_length 256 \
    --save_features_qa data/qa_ranking/npy \
