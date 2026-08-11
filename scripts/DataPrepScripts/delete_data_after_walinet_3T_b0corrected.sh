#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="$(realpath -- "${PROJECT_DIR}/data/3T/B0corrected_wo_LipidMask")"

if [[ ! -d "${DATA_ROOT}" ]]; then
    echo "Data directory not found: ${DATA_ROOT}" >&2
    exit 1
fi

# Expected layout below DATA_ROOT:
#   <scanner>/<subject>/<resolution>/OriginalData/data_after_walinet.npy
mapfile -d '' FILES < <(
    find "${DATA_ROOT}" \
        -mindepth 5 -maxdepth 5 \
        -type f \
        -path '*/OriginalData/data_after_walinet.npy' \
        -print0
)

if (( ${#FILES[@]} == 0 )); then
    echo "No data_after_walinet.npy files found below: ${DATA_ROOT}"
    exit 0
fi

echo "The following ${#FILES[@]} file(s) will be deleted:"
printf '  %s\n' "${FILES[@]}"
echo
du -ch -- "${FILES[@]}" | tail -n 1
echo

if [[ "${1:-}" != "--yes" ]]; then
    read -r -p "Permanently delete exactly these files? [y/N] " REPLY
    if [[ ! "${REPLY}" =~ ^[Yy]$ ]]; then
        echo "Cancelled; nothing was deleted."
        exit 0
    fi
fi

for FILE_PATH in "${FILES[@]}"; do
    rm -- "${FILE_PATH}"
done

echo "Deleted ${#FILES[@]} data_after_walinet.npy file(s)."
