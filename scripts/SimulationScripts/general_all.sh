#!/bin/bash
set -e

cd "$(dirname "$0")"

mkdir -p ../logs/logs_training_data
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="../logs/logs_training_data/generate_all_${timestamp}.log"

nohup bash -c '
  set -e
  bash generate_training_data_3T.sh
  bash generate_training_data_7T.sh
' > "$log_file" 2>&1 &

echo "Started sequential training-data generation."
echo "PID: $!"
echo "Log: $log_file"