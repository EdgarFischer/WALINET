#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/bs_PrismaMeasurementsForNNTraining/ThomShaw_CIMAXbrisbane"

DST_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/data/3T/ClimaX_Brisbane"

subjects=(
    "Vol4"
    "Vol5"
    "Vol6"
    "Vol7"
    "Vol8"
)

reconstructions=(
    "Res36x36"
    "Res50"
    "Res64thick"
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