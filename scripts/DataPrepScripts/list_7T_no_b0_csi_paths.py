#!/usr/bin/env python3
"""List stored csi_path values and report their aggregate existence status."""

from __future__ import annotations

import argparse
from pathlib import Path

from check_7T_no_b0_csi_paths import (
    DEFAULT_DATA_ROOT,
    checked_path,
    load_csi_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List the currently stored Par.Paths.csi_path value from every "
            "7T NoB0Correction CombinedCSI.mat file."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    mat_paths = sorted(data_root.rglob("OriginalData/CombinedCSI.mat"))
    if not mat_paths:
        raise SystemExit(f"No OriginalData/CombinedCSI.mat found below: {data_root}")

    existing = 0
    errors = 0

    for mat_path in mat_paths:
        # The subject/resolution directory immediately containing OriginalData,
        # shown relative to the configured branch for unambiguous output.
        folder_name = mat_path.parent.parent.relative_to(data_root)
        print(f"{folder_name}:")
        try:
            stored_path = load_csi_path(mat_path)
            print(stored_path)
            if checked_path(stored_path, mat_path).exists():
                existing += 1
        except (KeyError, TypeError, ValueError, OSError) as error:
            errors += 1
            print(f"<could not read Par.Paths.csi_path: {error}>")
        print()

    all_exist = existing == len(mat_paths) and errors == 0
    print(
        f"Alle Pfade existieren: {'JA' if all_exist else 'NEIN'} "
        f"({existing}/{len(mat_paths)})"
    )


if __name__ == "__main__":
    main()
