#!/bin/bash

python questgen/create_data/crawl_data/crawl_data.py \
    --domain "history" \
    --task "multiple-choice" \
    --source "chatgpt" \
    --data_path data/fschool/history_kg_crawl_data_v2.json \
    --accounts_path data/accounts.json \
    --database_config_path configs/crawling_database_config.yml \
    --is_preprocessing true
