#!/bin/bash
set -e

python questgen/inference/infer_aqg.py \
    --data_path data/mc/history/inference_qa.json \
    --task multitask \
    --multitask_model_name_or_path output/models/history/simple/v1.2 \
    --mc_model_name_or_path output/models/history/gcn/v1.2 \
    --config_aqg_path configs/faqg_pipeline_t5_vi_base_hl.yaml \
    --save_path_multitask output/simple/fschool_qa_simple_test_v1.1.json \
    --save_path_mc output/mc/fschool_mc_test_v1.0.4.json \
    --update_data False \
    --use_summary False \
    --use_multiprocess True \
    --only_distractors True \
