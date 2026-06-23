#!/bin/bash

SRC_BASE="/pfad/zur/quellstruktur"
DST_BASE="/pfad/zur/zielstruktur"

subjects=(
    "Vol01_WB/Res36x36"
    "Vol02_BS/Res36x36"
    "Vol03_SH/Res36x36"
    "Vol04_SM/Res36x36"
    "Vol05_LH/Res36x36"
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