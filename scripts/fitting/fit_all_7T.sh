#!/usr/bin/env bash
set -uo pipefail

WALINET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$WALINET_ROOT/.." && pwd)"

GPU_NUMBER="0"
PYTHON_BIN="${PYTHON:-/home/hfischer/venvs/walinet/bin/python}"

DATA_ROOT="$WALINET_ROOT/data/7T/B0corrected_wo_LipidMask"
MODEL_DIR="$WALINET_ROOT/models/7T_Final"
FORD_CONFIG="$WORKSPACE_ROOT/forD/runs/classical_fitting_config_trained Hauke.json"

LOG_DIR="$WALINET_ROOT/logs/logs_fitting"
TIMESTAMP="$(date +'%Y%m%d_%H%M%S')"
LOG_FILE="$LOG_DIR/fit_all_7T_${TIMESTAMP}.log"
LATEST_LOG="$LOG_DIR/fit_all_7T_latest.log"

mkdir -p "$LOG_DIR"
ln -sfn "$(basename "$LOG_FILE")" "$LATEST_LOG"
exec >>"$LOG_FILE" 2>&1

mapfile -d '' SUBJECT_DIRS < <(
    find -L "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z
)

echo "[$(date --iso-8601=seconds)] Starting WALINET + forD fitting"
echo "Subjects: ${#SUBJECT_DIRS[@]}"
echo "Data root: $DATA_ROOT"
echo "GPU: $GPU_NUMBER"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL_DIR"
echo "forD config: $FORD_CONFIG"
echo "Log: $LOG_FILE"
echo

successful=0
failed=0

for i in "${!SUBJECT_DIRS[@]}"; do
    SUBJECT_DIR="${SUBJECT_DIRS[$i]}"
    SUBJECT="$(basename "$SUBJECT_DIR")"

    DATA_PATH="$SUBJECT_DIR/OriginalData/data.npy"
    MASK_PATH="$SUBJECT_DIR/masks/brain_mask.npy"
    OUTPUT_DIR="$SUBJECT_DIR/MetabMapsAfterWalinet_Final"

    echo "================================================================"
    echo "[$(date --iso-8601=seconds)] Subject $((i + 1))/${#SUBJECT_DIRS[@]}: $SUBJECT"
    echo "Data: $DATA_PATH"
    echo "Mask: $MASK_PATH"
    echo "Output: $OUTPUT_DIR"
    echo

    if env \
        PYTHONPATH="$WALINET_ROOT/src:$WORKSPACE_ROOT/forD:$WORKSPACE_ROOT/mrs_utils${PYTHONPATH:+:$PYTHONPATH}" \
        "$PYTHON_BIN" -u -m walinet.inference.ford_pipeline \
        --data "$DATA_PATH" \
        --mask "$MASK_PATH" \
        --walinet-model "$MODEL_DIR" \
        --ford-config "$FORD_CONFIG" \
        --output "$OUTPUT_DIR" \
        --gpu "$GPU_NUMBER"
    then
        successful=$((successful + 1))
        echo "[$(date --iso-8601=seconds)] Finished: $SUBJECT"
    else
        failed=$((failed + 1))
        echo "[$(date --iso-8601=seconds)] FAILED: $SUBJECT"
    fi

    echo
done

echo "================================================================"
echo "[$(date --iso-8601=seconds)] All subjects processed"
echo "Successful: $successful"
echo "Failed: $failed"
echo "Total: ${#SUBJECT_DIRS[@]}"