#!/bin/bash

set -o errexit

if [[ -z "${PIPELINE_CONFIG_PATH}" ]]; then echo "Missing PIPELINE_CONFIG_PATH environment variable" >&2; exit 1; fi

set -o nounset

echo "Starting API server"
CUDA_VISIBLE_DEVICES=1 python3 restapi/main.py
