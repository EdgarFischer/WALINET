#!/usr/bin/env python3
"""Extract and cross-check stored Brisbane csi_path values without edits."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

from list_ucsf_parameter_csi_paths import load_csi_path


RESULTS_ROOT = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
    "LargeData_d3hj/Results_Dat_LCM_SameProcessingAsOnline/Brisbane"
)
VOLUME_DIRECTORY_PATTERN = re.compile(r"^Vol(0[2-5]|07)_Dat.*$")


def main() -> None:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {RESULTS_ROOT}")

    variants: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for directory in sorted(RESULTS_ROOT.iterdir()):
        if not directory.is_dir():
            continue
        match = VOLUME_DIRECTORY_PATTERN.fullmatch(directory.name)
        if match is None:
            continue

        parameters_path = directory / "Parameters.mat"
        if not parameters_path.is_file():
            raise FileNotFoundError(
                f"Expected Parameters.mat is missing: {parameters_path}"
            )
        variants[match.group(1)].append(
            (directory, load_csi_path(parameters_path))
        )

    if not variants:
        raise FileNotFoundError(f"No matching Vol*_Dat* directories in {RESULTS_ROOT}")

    conflicts = []
    for volume, entries in sorted(variants.items()):
        if len({path for _, path in entries}) > 1:
            details = "\n".join(
                f"    {directory.name}: {path}" for directory, path in entries
            )
            conflicts.append(f"Vol{volume} has different paths:\n{details}")

    if conflicts:
        print("ERROR: inconsistent paths found:\n", file=sys.stderr)
        print("\n\n".join(conflicts), file=sys.stderr)
        raise SystemExit(1)

    for volume, entries in sorted(variants.items()):
        print(f"Vol{volume}: {entries[0][1]}")

    all_found = all(Path(entries[0][1]).is_file() for entries in variants.values())
    print(f"Alle gespeicherten Pfade gefunden: {'JA' if all_found else 'NEIN'}")
    if not all_found:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
