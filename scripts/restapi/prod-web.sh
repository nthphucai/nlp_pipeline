#!/bin/bash

set -o errexit

if [[ -z "${GENERATE_ENDPOINT}" ]]; then echo "Missing ENDPOINT environment variable" >&2; exit 1; fi

set -o nounset

# echo "Starting streamlit"
# streamlit run app/main.py --server.port 5051

echo "Stating web demo"
python3 app/main.py