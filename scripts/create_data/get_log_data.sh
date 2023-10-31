#!/bin/bash

python3 dataset/create_data/modules/get_log_fromMongodb.py \
  --output_path data/fschool/output_log_data_v1.0.json \
  --connection_string mongodb://nlp:nlp2022@103.119.132.170:27017/?authMechanism=DEFAULT \
  --database_name aqg \
  --collection_name feedback
