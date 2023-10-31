#!/bin/bash
set -e

export $(cat restapi/.env | xargs)

python restapi/main.py
