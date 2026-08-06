#!/usr/bin/env bash
set -euo pipefail

WALINET_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BRANCH_ROOT="$WALINET_ROOT/data/3T/NoB0Correction"

mapfile -d '' TRAIN_DIRS < <(
  find -L "$BRANCH_ROOT" -type d -name TrainData -print0
)

if ((${#TRAIN_DIRS[@]} == 0)); then
  echo "No TrainData directories found under: $BRANCH_ROOT" >&2
  exit 1
fi

echo "Found ${#TRAIN_DIRS[@]} TrainData directories under: $BRANCH_ROOT"
du -sch "${TRAIN_DIRS[@]}"
