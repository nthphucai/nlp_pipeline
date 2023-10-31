#!/bin/bash
set -e

python create_dataset/parse_pdf/parse_pdf.py \
    --input_path ../../data/Textbooks/sach-giao-khoa-lich-su-11-preprocessed-split.pdf \
    --output_path ../../data/Textbooks/test.txt \