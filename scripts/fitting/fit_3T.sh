#!/usr/bin/env bash
set -euo pipefail

# Example: WALINET + classical forD fitting for 3T PRISMA Vienna Vol01_WB.
WALINET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$WALINET_ROOT/.." && pwd)"

GPU_NUMBER="${GPU_NUMBER:-0}"
PYTHON_BIN="${PYTHON:-/home/hfischer/venvs/walinet/bin/python}"
SUBJECT_DIR="$WALINET_ROOT/data/3T/B0corrected_wo_LipidMask/PRISMA_Vienna/Vol01_WB/Res50x50"
OUTPUT_DIR="$SUBJECT_DIR/MetabMapsAfterWalinet_2.1"
DATA_PATH="$SUBJECT_DIR/OriginalData/data.npy"
MASK_PATH="$SUBJECT_DIR/masks/brain_mask.npy"
LOG_DIR="$WALINET_ROOT/logs/logs_fitting"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/fit_3T_${TIMESTAMP}.log"
MODEL_FOLDER_NAME="3T_New_WALI_2.1"
MODEL_DIR="$WALINET_ROOT/models/$MODEL_FOLDER_NAME"
FORD_CONFIG="$WORKSPACE_ROOT/forD/runs/classical_fitting_config_Hauke_3T.json"

mkdir -p "$LOG_DIR"

{
  echo "[$(date --iso-8601=seconds)] Starting WALINET + forD fitting"
  echo "Data: $DATA_PATH"
  echo "Mask: $MASK_PATH"
  echo "GPU: $GPU_NUMBER"
  echo "Python: $PYTHON_BIN"
  echo "Model: $MODEL_DIR"
  echo "forD config: $FORD_CONFIG"
  echo "Output: $OUTPUT_DIR"
  echo
} >"$LOG_FILE"

nohup env \
  PYTHONPATH="$WALINET_ROOT/src:$WORKSPACE_ROOT/forD:$WORKSPACE_ROOT/mrs_utils${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -u -m walinet.inference.ford_pipeline \
  --data "$DATA_PATH" \
  --mask "$MASK_PATH" \
  --walinet-model "$MODEL_DIR" \
  --ford-config "$FORD_CONFIG" \
  --output "$OUTPUT_DIR" \
  --gpu "$GPU_NUMBER" \
  >>"$LOG_FILE" 2>&1 &

PID=$!
sleep 1
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Pipeline exited immediately. Check the log: $LOG_FILE" >&2
  exit 1
fi

echo "Started WALINET + forD fitting for: $DATA_PATH"
echo "PID: $PID"
echo "GPU: $GPU_NUMBER"
echo "Model: $MODEL_DIR"
echo "Output: $OUTPUT_DIR"
echo "Log: $LOG_FILE"
echo "Follow: tail -f $LOG_FILE"
