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

ALL_PATHS=(
  "${TRAIN_DIRS[@]}"
  "${B0_CORRECTED_FILES[@]}"
  "${SUPPRESSED_WATER_FILES[@]}"
)

if ((${#ALL_PATHS[@]} == 0)); then
  echo "Nothing to delete under: $BRANCH_ROOT"
  exit 0
fi

echo "Deletion targets under: $BRANCH_ROOT"
echo "  TrainData directories:       ${#TRAIN_DIRS[@]}"
echo "  data_B0corrected.npy files:   ${#B0_CORRECTED_FILES[@]}"
echo "  SupressedWater.npy files:     ${#SUPPRESSED_WATER_FILES[@]}"
echo "  Combined disk usage:          $(du -sch "${ALL_PATHS[@]}" | tail -n 1 | cut -f1)"
echo
printf '  %s\n' "${ALL_PATHS[@]}"

if [[ "${1:-}" != "--delete" ]]; then
  echo
  echo "Dry run only; nothing was deleted."
  echo "To permanently delete exactly these targets, run:"
  echo "  $0 --delete"
  exit 0
fi

echo
echo "WARNING: This permanently deletes the targets above; no trash or backup is used."
read -r -p "Type DELETE to continue: " CONFIRMATION
if [[ "$CONFIRMATION" != "DELETE" ]]; then
  echo "Cancelled; nothing was deleted."
  exit 1
fi

if ((${#TRAIN_DIRS[@]} > 0)); then
  rm -rf -- "${TRAIN_DIRS[@]}"
fi
if ((${#B0_CORRECTED_FILES[@]} > 0)); then
  rm -- "${B0_CORRECTED_FILES[@]}"
fi
if ((${#SUPPRESSED_WATER_FILES[@]} > 0)); then
  rm -- "${SUPPRESSED_WATER_FILES[@]}"
fi

echo "Deletion complete."
