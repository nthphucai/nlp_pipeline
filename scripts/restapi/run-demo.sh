#!/bin/bash
set -e

export $(cat app/.env | xargs)

streamlit run app/main.py --server.port 5050