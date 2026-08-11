#!/usr/bin/env python3
"""Print and validate Par.Paths.csi_path in all 7T NoB0 CombinedCSI files."""

from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path

import h5py
import mat73
import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "7T" / "NoB0Correction"
OLD_BRISBANE_ROOT = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/"
    "home/zeftekhari/Brisbane_Data"
)
NEW_BRISBANE_ROOT = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
    "LargeData_d3hj/MeasAndLogData/Brisbane"
)
STEP2_DATA_ROOT = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/"
    "Step2_ISMRMAbstractOnPipeline/LargeData_d3hj/MeasAndLogData"
)
STEP5_DATA_ROOT = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/"
    "Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData"
)
PATH_REPLACEMENTS = (
    (OLD_BRISBANE_ROOT, NEW_BRISBANE_ROOT),
    (
        f"{NEW_BRISBANE_ROOT}/MRSI-TEST-5/",
        f"{NEW_BRISBANE_ROOT}/MRSI-TEST-5/NotUsed/",
    ),
    (
        f"{STEP5_DATA_ROOT}/London/20250616_M701121_METAHEAD",
        f"{STEP5_DATA_ROOT}/London/Vol2_20250616_M701121_METAHEAD",
    ),
    (
        "20250612_M701118_METAHEAD",
        "Vol1_20250612_M701118_METAHEAD",
    ),
    (
        "20250620_M701126_METAHEAD",
        "Vol3_20250620_M701126_METAHEAD",
    ),
    (
        "20250620_M701128_METAHEAD",
        "Vol4_20250620_M701128_METAHEAD",
    ),
    (
        f"{STEP2_DATA_ROOT}/Vol5_Berni",
        f"{STEP5_DATA_ROOT}/Vienna/Vol5_Berni",
    ),
    (
        "20250626_M701130_METAHEAD",
        "Vol5_20250626_M701130_METAHEAD",
    ),
    (
        f"{STEP2_DATA_ROOT}/Vienna/Vol7_WolfgangB",
        f"{STEP5_DATA_ROOT}/Vienna/Vol7_WolfgangB",
    ),
    (
        f"{STEP2_DATA_ROOT}/Vol7_WolfgangB",
        f"{STEP5_DATA_ROOT}/Vienna/Vol7_WolfgangB",
    ),
    (
        f"{STEP2_DATA_ROOT}/Vol8_AnnaZ",
        f"{STEP5_DATA_ROOT}/Vienna/Vol8_AnnaZ",
    ),
)


def _unwrap_singleton(value):
    while True:
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
            continue
        if isinstance(value, (list, tuple)) and len(value) == 1:
            value = value[0]
            continue
        break
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


def _as_path_string(value) -> str:
    value = _unwrap_singleton(value)
    if isinstance(value, bytes):
        return value.decode()
    if isinstance(value, str):
        return value
    if isinstance(value, np.ndarray) and value.dtype.kind in {"U", "S"}:
        return "".join(str(item) for item in value.reshape(-1))
    raise TypeError(f"Expected a string path, got {type(value).__name__}.")


def load_csi_path(mat_path: Path) -> str:
    """Return the path string exactly as stored in Par.Paths.csi_path."""
    try:
        data = loadmat(
            mat_path,
            variable_names=["Par"],
            squeeze_me=True,
            struct_as_record=False,
        )
    except NotImplementedError:
        data = mat73.loadmat(
            mat_path,
            only_include=["Par"],
            verbose=False,
        )

    par = data["Par"]
    paths = _field(par, "Paths")
    csi_path = _as_path_string(_field(paths, "csi_path")).strip()
    if not csi_path:
        raise ValueError("Par.Paths.csi_path is empty.")
    return csi_path


def checked_path(stored_path: str, mat_path: Path) -> Path:
    """Expand a stored path and resolve relative paths beside CombinedCSI.mat."""
    path = Path(os.path.expandvars(stored_path)).expanduser()
    if not path.is_absolute():
        path = mat_path.parent / path
    return path.resolve()


def proposed_csi_path(stored_path: str) -> tuple[str, list[tuple[str, str]]]:
    """Apply every configured literal replacement and report those used."""
    result = stored_path
    applied = []
    for old, new in PATH_REPLACEMENTS:
        # Skip a rule whose final replacement is already present. This makes
        # repeated dry-runs and --apply invocations idempotent even when the
        # search text is itself a substring of the replacement text.
        if old in result and new not in result:
            result = result.replace(old, new)
            applied.append((old, new))
    return result, applied


def replace_hdf5_csi_path(mat_path: Path, new_path: str) -> None:
    """Replace the referenced MATLAB-v7.3 char array without rewriting the MAT."""
    code_points = np.asarray([ord(character) for character in new_path], dtype=np.uint16)
    if any(ord(character) > np.iinfo(np.uint16).max for character in new_path):
        raise ValueError("The replacement path contains unsupported Unicode characters.")

    with h5py.File(mat_path, "r+") as mat_file:
        path_cell = mat_file["Par/Paths/csi_path"]
        if path_cell.size != 1 or h5py.check_dtype(ref=path_cell.dtype) is None:
            raise TypeError(
                "Par/Paths/csi_path is not the expected singleton MATLAB cell reference."
            )

        references = mat_file.require_group("#refs#")
        dataset_name = f"walinet_csi_path_{uuid.uuid4().hex}"
        path_dataset = references.create_dataset(
            dataset_name,
            data=code_points.reshape(-1, 1),
            dtype=np.uint16,
        )
        path_dataset.attrs["H5PATH"] = np.bytes_(f"/#refs#/{dataset_name}")
        path_dataset.attrs["MATLAB_class"] = np.bytes_("char")
        path_dataset.attrs["MATLAB_int_decode"] = np.int32(2)
        path_cell[0, 0] = path_dataset.ref
        mat_file.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Par.Paths.csi_path in every OriginalData/CombinedCSI.mat "
            "below the 7T NoB0Correction data branch."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the configured path replacements into matching MATLAB-v7.3 "
            "files. Without this flag the script is read-only."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    mat_paths = sorted(data_root.rglob("OriginalData/CombinedCSI.mat"))
    if not mat_paths:
        raise SystemExit(f"No OriginalData/CombinedCSI.mat found below: {data_root}")

    print(f"Data root: {data_root}")
    print(f"CombinedCSI files: {len(mat_paths)}")
    print(f"Mode: {'APPLY' if args.apply else 'DRY RUN'}")
    print("Configured replacements:")
    for old, new in PATH_REPLACEMENTS:
        print(f"  {old} -> {new}")
    print()

    valid = 0
    warnings = 0
    replacements = 0
    for index, mat_path in enumerate(mat_paths, start=1):
        relative_mat_path = mat_path.relative_to(data_root)
        print(f"[{index:03d}/{len(mat_paths):03d}] {relative_mat_path}")
        try:
            stored_path = load_csi_path(mat_path)
            print(f"  Par.Paths.csi_path: {stored_path}")

            effective_path, applied_rules = proposed_csi_path(stored_path)
            if applied_rules:
                print(f"  Replacement:       {effective_path}")
                for old, new in applied_rules:
                    print(f"    rule: {old} -> {new}")
                if args.apply:
                    if not h5py.is_hdf5(mat_path):
                        raise TypeError(
                            "Prefix replacement is currently supported only for "
                            "MATLAB-v7.3/HDF5 CombinedCSI files."
                        )
                    replace_hdf5_csi_path(mat_path, effective_path)
                    verified_path = load_csi_path(mat_path)
                    if verified_path != effective_path:
                        raise RuntimeError(
                            "Written csi_path failed read-back verification: "
                            f"{verified_path}"
                        )
                    print("  Update: APPLIED AND VERIFIED")
                else:
                    print("  Update: DRY RUN (use --apply to write)")
                replacements += 1

            resolved_path = checked_path(effective_path, mat_path)
            # csi_path may reference either a Siemens .dat file or a directory
            # containing reconstructed spectral data, depending on the site.
            if resolved_path.exists():
                valid += 1
                print("  Status: OK")
            else:
                warnings += 1
                print(f"  WARNING: referenced path does not exist: {resolved_path}")
        except (KeyError, TypeError, ValueError, OSError) as error:
            warnings += 1
            print(f"  WARNING: could not read Par.Paths.csi_path: {error}")
        print()

    print(
        f"Summary: {len(mat_paths)} CombinedCSI file(s), "
        f"{valid} existing csi_path target(s), {warnings} warning(s), "
        f"{replacements} file replacement(s) "
        f"{'applied' if args.apply else 'proposed'}."
    )


if __name__ == "__main__":
    main()
