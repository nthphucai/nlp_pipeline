#!/bin/bash

set -o errexit

echo "One-click Development"
docker-compose -f docker/docker-compose.prod.yml down
docker build -t metanlp/questgen-api:v1.0 -f docker/Dockerfile.api .
docker build -t metanlp/questgen:v1.0 -f docker/Dockerfile .
docker-compose -f docker/docker-compose.prod.yml up -d
