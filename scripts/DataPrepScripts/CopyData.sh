#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/bs_PrismaMeasurementsForNNTraining/PRISMAFIT"

DST_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/data/3T/PRISMAFIT_Vienna"

subjects=(
    "Vol01_BS"
    "Vol02_LH"
    "Vol03_WB"
    "Vol04_LP"
)

reconstructions=(
    "36"
    "50"
    "64"
)

for subject in "${subjects[@]}"; do
    for reconstruction in "${reconstructions[@]}"; do

        src_dir="${SRC_BASE}/${subject}/${reconstruction}"
        dst_dir="${DST_BASE}/${subject}/${reconstruction}/OriginalData"

        echo "Processing ${subject}/${reconstruction}"

        if [[ ! -d "${src_dir}" ]]; then
            echo "ERROR: Source directory does not exist: ${src_dir}" >&2
            exit 1
        fi

        if [[ ! -f "${src_dir}/CombinedCSI.mat" ]]; then
            echo "ERROR: Missing CombinedCSI.mat: ${src_dir}/CombinedCSI.mat" >&2
            exit 1
        fi

        if [[ ! -d "${src_dir}/maps" ]]; then
            echo "ERROR: Missing maps directory: ${src_dir}/maps" >&2
            exit 1
        fi

        mkdir -p "${dst_dir}"

        cp "${src_dir}/CombinedCSI.mat" "${dst_dir}/"
        cp -r "${src_dir}/maps" "${dst_dir}/"

    done

    echo "Successfully finished ${subject}"
done

echo "Done."