#!/bin/bash
set -e

python questgen/trainer/run_rank.py \
    --train_data_path data/qa_ranking/json/v1.3/train_data_v1.3.json \
    --eval_data_path data/qa_ranking/json/v1.3/dev_data_v1.3.json \
    --gaussian_mixture_path data/qa_ranking/gaussian_mixture.dump \
    --isotonic_regressor_path data/qa_ranking/isotonic_regressor.dump \
    --features_qa_path data/qa_ranking/npy \
