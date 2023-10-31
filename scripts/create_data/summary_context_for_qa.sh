#!/bin/bash
set -e

python create_dataset/extract_context/create_dataset.py \
    --task "summary" \
    --qa_pair_data_path ../../data/fschool/QA_LS11_LS12_Data.csv \
    --context_data ../../data/fschool/history_11-12_textbooks_content.txt \
    --SAQG_PATH ../../output/fschool/SAQG_test_module.json \
    --MCQG_PATH ../../output/fschool/MCQG_test_module.json \
    --SAQGSUM_PATH ../../output/fschool/SAQG_test_module_summary.json \
    --MCQGSUM_PATH ../../output/fschool/MCAG_test_module_summary.json \
    # --use_multiprocessing True \
