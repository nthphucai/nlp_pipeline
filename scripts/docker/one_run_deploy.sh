#!/bin/bash

set -o errexit

echo "One-click Development"
docker-compose -f docker/docker-compose.prod.yml up -d
