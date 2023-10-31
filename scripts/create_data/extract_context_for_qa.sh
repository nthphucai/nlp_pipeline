#!/bin/bash
set -e

python questgen/dataset/create_data_from_textbook.py \
    --task "extract-context-bm25" \
    --qa_pair_data_path data/fschool/text-book/QA_LS11_LS12_Data.csv \
    --context_data data/fschool/text-book/history_11-12_textbooks_content.txt \
    --AQG_PATH output/fschool/SAQG_test_module.json \
    --AQGSUM_PATH output/fschool/SAQG_test_module_summary.json \
    --use_multiprocessing True \
    --num_workers 20 \
