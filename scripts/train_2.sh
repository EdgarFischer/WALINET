#!/bin/bash
set -e

cd "$(dirname "$0")/.."

mkdir -p logs/logs_training

timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="logs/logs_training/train_${timestamp}.log"

nohup python3 -u scripts/train.py \
  --config configs/train_2.yaml \
  > "$log_file" 2>&1 &

echo "Started training."
echo "PID: $!"
echo "Log: $log_file"