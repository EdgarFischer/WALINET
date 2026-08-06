#!/usr/bin/env bash
set -euo pipefail

WALINET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH_ROOT="$WALINET_ROOT/data/3T/NoB0Correction"

mapfile -d '' TRAIN_DIRS < <(
  find -L "$BRANCH_ROOT" -type d -name TrainData -print0
)

if ((${#TRAIN_DIRS[@]} == 0)); then
  echo "No TrainData directories found under: $BRANCH_ROOT"
  exit 0
fi

echo "Deletion targets under: $BRANCH_ROOT"
echo "  TrainData directories: ${#TRAIN_DIRS[@]}"
echo "  Combined disk usage:    $(du -sch "${TRAIN_DIRS[@]}" | tail -n 1 | cut -f1)"
echo
printf '  %s\n' "${TRAIN_DIRS[@]}"

if [[ "${1:-}" != "--delete" ]]; then
  echo
  echo "Dry run only; nothing was deleted."
  echo "To permanently delete exactly these TrainData directories, run:"
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

rm -rf -- "${TRAIN_DIRS[@]}"

echo "Deletion complete."
