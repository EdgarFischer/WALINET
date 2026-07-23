#!/usr/bin/env bash

set -euo pipefail

base_dir="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/3T"

# Ausgabe im selben Ordner wie dieses Skript
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
output_file="$script_dir/subjects_3T.txt"

if [[ ! -d "$base_dir" ]]; then
    echo "Fehler: Base directory existiert nicht: $base_dir" >&2
    exit 1
fi

mapfile -t subjects < <(
    find "$base_dir" \
        -mindepth 3 \
        -maxdepth 3 \
        -type d \
        \( \
            -name "Res36x36" \
            -o -name "Res50x50" \
            -o -name "Res64x64x41" \
            -o -name "Res64x64x47" \
        \) \
        -printf '%P\n' |
        sort -V
)

if [[ ${#subjects[@]} -eq 0 ]]; then
    echo "Keine passenden Resolution-Ordner gefunden." >&2
    exit 1
fi

{
    printf '  subjects:\n'

    current_group=""

    for path in "${subjects[@]}"; do
        # Alles vor dem ersten "/" ist der Oberordner
        group="${path%%/*}"

        # Bei einem neuen Oberordner Kommentar und Leerzeile einfügen
        if [[ "$group" != "$current_group" ]]; then
            if [[ -n "$current_group" ]]; then
                printf '\n'
            fi

            printf '    # %s\n' "$group"
            current_group="$group"
        fi

        # Sonderzeichen für YAML-Strings absichern
        escaped_path="${path//\\/\\\\}"
        escaped_path="${escaped_path//\"/\\\"}"

        printf '    - "%s"\n' "$escaped_path"
    done
} > "$output_file"

echo "Liste gespeichert unter:"
echo "$output_file"