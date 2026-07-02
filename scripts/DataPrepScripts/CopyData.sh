#!/bin/bash

SRC_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/kchaknam/Berni_pipeline/output/MS_180"
DST_BASE="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/data/7T/NoB0Correction"

subjects=(
    "3D-CRT_m64x64x33_0p2_3p88ppm_woL2"    
)

for subject in "${subjects[@]}"; do

    src_dir="${SRC_BASE}/${subject}"
    dst_dir="${DST_BASE}/${subject}/OriginalData"

    echo "Processing ${subject}"

    mkdir -p "${dst_dir}"

    cp "${src_dir}/CombinedCSI.mat" "${dst_dir}/"
    cp -r "${src_dir}/maps" "${dst_dir}/"

done

echo "Done."
