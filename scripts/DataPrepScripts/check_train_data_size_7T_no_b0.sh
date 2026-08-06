#!/usr/bin/env bash
set -euo pipefail

WALINET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH_ROOT="$WALINET_ROOT/data/7T/NoB0Correction"

mapfile -d '' TRAIN_DIRS < <(
  find -L "$BRANCH_ROOT" -mindepth 2 -maxdepth 2 -type d -name TrainData -print0
)
mapfile -d '' B0_CORRECTED_FILES < <(
  find -L "$BRANCH_ROOT" -mindepth 3 -maxdepth 3 -type f \
    -path '*/OriginalData/data_B0corrected.npy' -print0
)
mapfile -d '' SUPPRESSED_WATER_FILES < <(
  find -L "$BRANCH_ROOT" -mindepth 3 -maxdepth 3 -type f \
    -path '*/OriginalData/SupressedWater.npy' -print0
)

if ((${#TRAIN_DIRS[@]} == 0)); then
  echo "No TrainData directories found under: $BRANCH_ROOT" >&2
  exit 1
fi

echo "Found ${#TRAIN_DIRS[@]} TrainData directories under: $BRANCH_ROOT"
du -sch "${TRAIN_DIRS[@]}"

echo
echo "Found ${#B0_CORRECTED_FILES[@]} data_B0corrected.npy files"
if ((${#B0_CORRECTED_FILES[@]} > 0)); then
  du -sch "${B0_CORRECTED_FILES[@]}"
fi

echo
echo "Found ${#SUPPRESSED_WATER_FILES[@]} SupressedWater.npy files"
if ((${#SUPPRESSED_WATER_FILES[@]} > 0)); then
  du -sch "${SUPPRESSED_WATER_FILES[@]}"
fi

ALL_PATHS=(
  "${TRAIN_DIRS[@]}"
  "${B0_CORRECTED_FILES[@]}"
  "${SUPPRESSED_WATER_FILES[@]}"
)

echo
echo "Combined total (TrainData + data_B0corrected.npy + SupressedWater.npy):"
du -sch "${ALL_PATHS[@]}" | tail -n 1
