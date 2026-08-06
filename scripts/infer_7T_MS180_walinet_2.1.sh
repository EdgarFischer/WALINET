#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PYTHON_BIN="/home/hfischer/venvs/walinet/bin/python"
GPU_NUMBER="1"
BATCH_SIZE="200"

DATA_PATH="${PROJECT_ROOT}/data/7T/B0corrected_wo_LipidMask/MS_180/OriginalData/data.npy"
MASK_PATH="${PROJECT_ROOT}/data/7T/B0corrected_wo_LipidMask/MS_180/masks/brain_mask.npy"
MODEL_DIR="${PROJECT_ROOT}/models/7T_New_WALI_2.1"
OUTPUT_PATH="${PROJECT_ROOT}/data/7T/B0corrected_wo_LipidMask/MS_180/OriginalData/data_after_walinet.npy"
DENOISING_OUTPUT_PATH="${PROJECT_ROOT}/../Denoising/datasets/Proton/7T/B0corrected_wo_LipidMask/MS_180/OriginalData/data_after_walinet.npy"
TEMP_OUTPUT_PATH="${OUTPUT_PATH%.npy}.WALINET_2.1.tmp.npy"

LOG_DIR="${PROJECT_ROOT}/logs/inference"
LOG_FILE="${LOG_DIR}/infer_7T_MS180_WALINET_2.1.log"
PID_FILE="${LOG_DIR}/infer_7T_MS180_WALINET_2.1.pid"

mkdir -p "${LOG_DIR}"

for required_path in "${PYTHON_BIN}" "${DATA_PATH}" "${MASK_PATH}" "${MODEL_DIR}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "Required path not found: ${required_path}" >&2
        exit 1
    fi
done

if [[ -e "${TEMP_OUTPUT_PATH}" ]]; then
    echo "Temporary output already exists; refusing to overwrite it:" >&2
    echo "${TEMP_OUTPUT_PATH}" >&2
    exit 1
fi

{
    echo "[$(date --iso-8601=seconds)] Starting WALINET 2.1 inference"
    echo "Data:       ${DATA_PATH}"
    echo "Mask:       ${MASK_PATH}"
    echo "Model:      ${MODEL_DIR}"
    echo "Checkpoint: model_best.pt"
    echo "GPU:        ${GPU_NUMBER}"
    echo "Batch size: ${BATCH_SIZE}"
    echo "Temporary:  ${TEMP_OUTPUT_PATH}"
    echo "Output:     ${OUTPUT_PATH}"
    echo "Denoising:  ${DENOISING_OUTPUT_PATH}"
    echo
} > "${LOG_FILE}"

nohup env \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
    DATA_PATH="${DATA_PATH}" \
    MASK_PATH="${MASK_PATH}" \
    MODEL_DIR="${MODEL_DIR}" \
    OUTPUT_PATH="${OUTPUT_PATH}" \
    DENOISING_OUTPUT_PATH="${DENOISING_OUTPUT_PATH}" \
    TEMP_OUTPUT_PATH="${TEMP_OUTPUT_PATH}" \
    GPU_NUMBER="${GPU_NUMBER}" \
    BATCH_SIZE="${BATCH_SIZE}" \
    "${PYTHON_BIN}" -u -c '
import os
import shutil
from pathlib import Path

import numpy as np

from walinet.inference.fid_inference import infer_fid

data_path = Path(os.environ["DATA_PATH"])
mask_path = Path(os.environ["MASK_PATH"])
model_dir = Path(os.environ["MODEL_DIR"])
output_path = Path(os.environ["OUTPUT_PATH"])
denoising_output_path = Path(os.environ["DENOISING_OUTPUT_PATH"])
temporary_path = Path(os.environ["TEMP_OUTPUT_PATH"])
device = "cuda:" + os.environ["GPU_NUMBER"]
batch_size = int(os.environ["BATCH_SIZE"])

source = np.load(data_path, mmap_mode="r", allow_pickle=False)
expected_shape = source.shape
print(f"[inference] Input shape={expected_shape}, dtype={source.dtype}", flush=True)
del source

infer_fid(
    data_path,
    model_dir,
    output_path=temporary_path,
    fid_axis=3,
    headmask=mask_path,
    checkpoint="model_best.pt",
    batch_size=batch_size,
    device=device,
)

candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
if candidate.shape != expected_shape:
    raise RuntimeError(
        f"Output shape {candidate.shape} does not match input {expected_shape}."
    )
if candidate.dtype != np.complex64:
    raise RuntimeError(f"Expected complex64 output, got {candidate.dtype}.")
if not np.isfinite(candidate).all():
    raise RuntimeError("Inference output contains non-finite values.")
print(
    f"[inference] Validated temporary output: shape={candidate.shape}, "
    f"dtype={candidate.dtype}",
    flush=True,
)
del candidate

# Copying into an existing destination preserves its inode and therefore the
# WALINET/Denoising hardlink. The old file is touched only after validation.
shutil.copyfile(temporary_path, output_path)
if (
    not denoising_output_path.exists()
    or os.stat(output_path).st_ino != os.stat(denoising_output_path).st_ino
):
    shutil.copyfile(temporary_path, denoising_output_path)

temporary_path.unlink()
print(f"[inference] Updated WALINET output: {output_path}", flush=True)
print(f"[inference] Updated Denoising input: {denoising_output_path}", flush=True)
print("[inference] Finished successfully.", flush=True)
' >> "${LOG_FILE}" 2>&1 &

PID=$!
echo "${PID}" > "${PID_FILE}"

sleep 1
if kill -0 "${PID}" 2>/dev/null; then
    echo "WALINET inference started with PID ${PID}"
    echo "Log: ${LOG_FILE}"
    echo "Follow: tail -f ${LOG_FILE}"
else
    echo "Inference stopped during startup. Check the log:" >&2
    echo "${LOG_FILE}" >&2
    exit 1
fi
