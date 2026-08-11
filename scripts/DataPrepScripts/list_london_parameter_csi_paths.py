#!/usr/bin/env python3
"""Extract and cross-check Par.Paths.csi_path from London Parameters.mat files."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import mat73
import numpy as np
from scipy.io import loadmat


DEFAULT_RESULTS_ROOT = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
    "LargeData_d3hj/Results_Dat_LCM_SameProcessingAsOnline/London"
)
VOLUME_DIRECTORY_PATTERN = re.compile(r"^Vol(0[1-5])_Dat.*$")
SUBJECT_DIRECTORY_REPLACEMENTS = {
    "01": ("20250612_M701118_METAHEAD", "Vol1_20250612_M701118_METAHEAD"),
    "02": ("20250616_M701121_METAHEAD", "Vol2_20250616_M701121_METAHEAD"),
    "03": ("20250620_M701126_METAHEAD", "Vol3_20250620_M701126_METAHEAD"),
    "04": ("20250620_M701128_METAHEAD", "Vol4_20250620_M701128_METAHEAD"),
    "05": ("20250626_M701130_METAHEAD", "Vol5_20250626_M701130_METAHEAD"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read Par.Paths.csi_path from every immediate Vol01_Dat* through "
            "Vol05_Dat* subdirectory. Multiple variants of one volume must "
            "contain exactly the same path."
        )
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help=f"London results directory (default: {DEFAULT_RESULTS_ROOT})",
    )
    return parser.parse_args()


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
    """Read Par.Paths.csi_path from classic or MATLAB-v7.3 Parameters.mat."""
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


def corrected_csi_path(volume: str, stored_path: str) -> str:
    """Return the volume-specific corrected London acquisition path."""
    old, new = SUBJECT_DIRECTORY_REPLACEMENTS[volume]
    if new in stored_path:
        return stored_path
    if old not in stored_path:
        raise ValueError(
            f"Vol{volume}: expected path component {old!r} was not found in "
            f"Par.Paths.csi_path: {stored_path}"
        )
    return stored_path.replace(old, new)


def main() -> None:
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    if not results_root.is_dir():
        raise FileNotFoundError(f"Results directory does not exist: {results_root}")

    variants: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    for directory in sorted(results_root.iterdir()):
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
            (
                directory,
                corrected_csi_path(
                    match.group(1),
                    load_csi_path(parameters_path),
                ),
            )
        )

    if not variants:
        raise FileNotFoundError(
            f"No Vol01_Dat* through Vol05_Dat* directories found in {results_root}"
        )

    conflicts = []
    for volume, entries in sorted(variants.items()):
        unique_paths = {path for _, path in entries}
        if len(unique_paths) > 1:
            details = "\n".join(
                f"    {directory.name}: {path}" for directory, path in entries
            )
            conflicts.append(f"Vol{volume} contains different csi_path values:\n{details}")

    if conflicts:
        print("ERROR: inconsistent paths found:\n", file=sys.stderr)
        print("\n\n".join(conflicts), file=sys.stderr)
        raise SystemExit(1)

    for volume, entries in sorted(variants.items()):
        print(f"Vol{volume}: {entries[0][1]}")

    missing_paths = [
        (volume, Path(entries[0][1]))
        for volume, entries in sorted(variants.items())
        if not Path(entries[0][1]).is_file()
    ]

    if missing_paths:
        print("Alle korrigierten Pfade gefunden: NEIN", file=sys.stderr)
        raise SystemExit(1)

    print("Alle korrigierten Pfade gefunden: JA")


if __name__ == "__main__":
    main()
