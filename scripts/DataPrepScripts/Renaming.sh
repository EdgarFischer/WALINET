#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/data/3T/VIDA_Vienna"

subjects=(
    #"Vol01_PW"
    "Vol02_BS"
    "Vol03_TE"
    "Vol04_AA"
    "Vol05_LH"
)

old_names=(
    "36"
    "50"
    "64"
)

new_names=(
    "Res36x36"
    "Res50x50"
    "Res64x64x41"
)

# Sicherheitscheck
if [[ ${#old_names[@]} -ne ${#new_names[@]} ]]; then
    echo "ERROR: old_names and new_names must have the same length." >&2
    exit 1
fi

for subject in "${subjects[@]}"; do
    for ((i=0; i<${#old_names[@]}; i++)); do

        old_dir="${BASE_DIR}/${subject}/${old_names[$i]}"
        new_dir="${BASE_DIR}/${subject}/${new_names[$i]}"

        if [[ ! -d "${old_dir}" ]]; then
            echo "WARNING: ${old_dir} does not exist. Skipping."
            continue
        fi

        if [[ -e "${new_dir}" ]]; then
            echo "ERROR: Target already exists: ${new_dir}" >&2
            exit 1
        fi

        mv "${old_dir}" "${new_dir}"

        echo "Renamed: ${subject}/${old_names[$i]} -> ${new_names[$i]}"

    done
done

echo "Done."