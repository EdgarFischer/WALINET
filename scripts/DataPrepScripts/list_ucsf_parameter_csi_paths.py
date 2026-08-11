#!/usr/bin/env python3
"""Extract and cross-check stored UCSF csi_path values without modifying them."""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import mat73
import numpy as np
from scipy.io import loadmat


RESULTS_ROOT = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
    "LargeData_d3hj/Results_Dat_LCM_SameProcessingAsOnline/UCSF"
)
VOLUME_DIRECTORY_PATTERN = re.compile(r"^Vol(0[1-5])_Dat.*$")


def _unwrap_singleton(value):
    while True:
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        return value


def _field(value, name: str):
    value = _unwrap_singleton(value)
    if isinstance(value, dict):
        return value[name]
    if isinstance(value, np.void) and value.dtype.names and name in value.dtype.names:
        return value[name]
    if hasattr(value, name):
        return getattr(value, name)
    raise KeyError(name)


def _as_string(value) -> str:
    value = _unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in value.reshape(-1))
    raise TypeError(f"Expected a string, got {type(value).__name__}.")


def load_csi_path(parameters_path: Path) -> str:
    try:
        content = loadmat(
            parameters_path,
            variable_names=["Par"],
            squeeze_me=True,
            struct_as_record=False,
        )
    except NotImplementedError:
        content = mat73.loadmat(
            parameters_path,
            only_include=["Par"],
            verbose=False,
        )

    path = _as_string(
        _field(_field(content["Par"], "Paths"), "csi_path")
    ).strip()
    if not path:
        raise ValueError("Par.Paths.csi_path is empty.")
    return path


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
