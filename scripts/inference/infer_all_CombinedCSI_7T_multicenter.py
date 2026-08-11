#!/usr/bin/env python3
"""Sequential B0 correction and WALINET inference for 20 multicenter scans."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_combined_csi  # noqa: E402


RESULTS_ROOT = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
    "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
    "LargeData_d3hj/Results_Dicom_LCM"
)
MODEL_DIR = PROJECT_ROOT / "models" / "7T_Final"
JULIA_EXECUTABLE = (
    PROJECT_ROOT / "B0_correction" / "julia-1.11.1" / "bin" / "julia"
)
JULIA_PROJECT = PROJECT_ROOT / "B0_correction"
OUTPUT_FILENAME = "CombinedCSI_B0corrected_after_WALINET.mat"
SUBJECT_PATTERN = re.compile(r"^Vol(\d{2})_DicomNew$")


DAT_PATHS = {
    ("London", "01"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol1_20250612_M701118_METAHEAD/meas_MID00109_FID32131_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",
    ("London", "02"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol2_20250616_M701121_METAHEAD/meas_MID00087_FID32273_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",
    ("London", "03"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol3_20250620_M701126_METAHEAD/meas_MID00086_FID32881_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",
    ("London", "04"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol4_20250620_M701128_METAHEAD/meas_MID00162_FID32957_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",
    ("London", "05"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol5_20250626_M701130_METAHEAD/meas_MID00099_FID33074_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",
    ("Vienna", "05"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol5_Berni/meas_MID00025_FID27701_csi_fidesi_crt_Feb2025_2.dat",
    ("Vienna", "06"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol6_LukasH/meas_MID00093_FID30255_csi_fidesi_crt_Feb2025_2.dat",
    ("Vienna", "07"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol7_WolfgangB/meas_MID00044_FID35273_csi_fidesi_crt_Feb2025_2.dat",
    ("Vienna", "08"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol8_AnnaZ/meas_MID00166_FID35660_csi_fid_ViennaCrt_v1_00.dat",
    ("Vienna", "09"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol9_StanoM/meas_MID00054_FID38080_csi_fidesi_crt_Feb2025_2.dat",
    ("UCSF", "01"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer1_20250819/meas_MID00229_FID32904_csi_fid_ViennaCrt_v1_01.dat",
    ("UCSF", "02"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer2_20250827/meas_MID00032_FID33306_csi_fid_ViennaCrt_v1_01.dat",
    ("UCSF", "03"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer3_20250827/meas_MID00068_FID33342_csi_fid_ViennaCrt_v1_01.dat",
    ("UCSF", "04"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer4_20250912/meas_MID00100_FID33703_csi_fid_ViennaCrt_v1_01.dat",
    ("UCSF", "05"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer5_20250912/meas_MID00033_FID33731_csi_fid_ViennaCrt_v1_01.dat",
    ("Brisbane", "02"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-2/meas_MID00166_FID04729_csi_fid_ViennaCrt_v1_01.dat",
    ("Brisbane", "03"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-3/meas_MID00035_FID06136_csi_fid_ViennaCrt_v1_01.dat",
    ("Brisbane", "04"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-4/meas_MID00039_FID06170_csi_fid_ViennaCrt_v1_01.dat",
    ("Brisbane", "05"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-5/meas_MID00177_FID07765_csi_fid_ViennaCrt_v1_01.dat",
    ("Brisbane", "07"): "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-7/meas_MID00033_FID13892_csi_fid_ViennaCrt_v1_01.dat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential B0 correction and 7T_Final WALINET inference."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default="model_last.pt")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=f"Replace existing {OUTPUT_FILENAME} files.",
    )
    return parser.parse_args()


def discover_inputs() -> dict[tuple[str, str], Path]:
    discovered = {}
    for site_dir in sorted(RESULTS_ROOT.iterdir()):
        if not site_dir.is_dir():
            continue
        for subject_dir in sorted(site_dir.iterdir()):
            if not subject_dir.is_dir():
                continue
            match = SUBJECT_PATTERN.fullmatch(subject_dir.name)
            if match is None:
                continue
            input_path = subject_dir / "CombinedCSI.mat"
            if not input_path.is_file():
                raise FileNotFoundError(f"CombinedCSI.mat is missing: {input_path}")
            key = (site_dir.name, match.group(1))
            if key in discovered:
                raise RuntimeError(f"Duplicate input for {key}: {subject_dir}")
            discovered[key] = input_path
    return discovered


def preflight(args: argparse.Namespace) -> list[tuple[tuple[str, str], Path, Path]]:
    if not RESULTS_ROOT.is_dir():
        raise FileNotFoundError(f"Results root does not exist: {RESULTS_ROOT}")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {MODEL_DIR}")
    if not (MODEL_DIR / args.checkpoint).is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {MODEL_DIR / args.checkpoint}")
    if not JULIA_EXECUTABLE.is_file():
        raise FileNotFoundError(f"Julia executable does not exist: {JULIA_EXECUTABLE}")
    if not JULIA_PROJECT.is_dir():
        raise FileNotFoundError(f"Julia project does not exist: {JULIA_PROJECT}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    discovered = discover_inputs()
    expected_keys = set(DAT_PATHS)
    discovered_keys = set(discovered)
    if discovered_keys != expected_keys:
        missing = sorted(expected_keys - discovered_keys)
        unexpected = sorted(discovered_keys - expected_keys)
        raise RuntimeError(
            "Input/mapping mismatch. "
            f"Missing CombinedCSI keys: {missing}; unexpected keys: {unexpected}"
        )
    if len(discovered) != 20:
        raise RuntimeError(f"Expected exactly 20 subjects, found {len(discovered)}.")

    jobs = []
    for key in sorted(discovered):
        dat_path = Path(DAT_PATHS[key])
        if not dat_path.is_file():
            raise FileNotFoundError(f"Siemens .dat file does not exist for {key}: {dat_path}")
        jobs.append((key, discovered[key], dat_path))
    return jobs


def main() -> None:
    args = parse_args()
    jobs = preflight(args)

    existing_outputs = [
        input_path.with_name(OUTPUT_FILENAME)
        for _, input_path, _ in jobs
        if input_path.with_name(OUTPUT_FILENAME).exists()
    ]
    if existing_outputs and not args.overwrite and not args.dry_run:
        paths = "\n".join(f"  {path}" for path in existing_outputs)
        raise FileExistsError(
            "Output files already exist; no inference was started. "
            "Use --overwrite to replace them:\n"
            f"{paths}"
        )

    print(f"Model:      {MODEL_DIR}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"Subjects:   {len(jobs)}", flush=True)
    print(f"Overwrite:  {args.overwrite}", flush=True)
    print(flush=True)

    for index, (key, input_path, dat_path) in enumerate(jobs, start=1):
        site, volume = key
        output_path = input_path.with_name(OUTPUT_FILENAME)
        print(f"[{index:02d}/{len(jobs):02d}] {site}/Vol{volume}", flush=True)
        print(f"  Input:  {input_path}", flush=True)
        print(f"  DAT:    {dat_path}", flush=True)
        print(f"  Output: {output_path}", flush=True)

        if output_path.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )
        if args.dry_run:
            print("  DRY RUN", flush=True)
            continue

        infer_combined_csi(
            input_path=input_path,
            model_dir=MODEL_DIR,
            output_path=output_path,
            checkpoint=args.checkpoint,
            batch_size=args.batch_size,
            device=args.device,
            b0_correction=True,
            dat_path=dat_path,
            julia_executable=JULIA_EXECUTABLE,
            julia_project=JULIA_PROJECT,
            shm_dir="/dev/shm",
        )
        print("  DONE", flush=True)

    if args.dry_run:
        print(f"Dry run completed: all {len(jobs)} jobs validated.", flush=True)
    else:
        print(f"Completed all {len(jobs)} subjects successfully.", flush=True)


if __name__ == "__main__":
    main()
