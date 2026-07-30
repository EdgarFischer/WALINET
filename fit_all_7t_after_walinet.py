#!/usr/bin/env python3
"""Sequentially fit every 7-T ``data_after_walinet.npy`` subject volume."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


WALINET_ROOT = Path(__file__).resolve().parent
WORKSPACE_ROOT = WALINET_ROOT.parent
DEFAULT_DATA_ROOT = WALINET_ROOT / "data/7T/B0corrected_wo_LipidMask"
DEFAULT_CONFIG = WORKSPACE_ROOT / "forD/runs/fitting_config_hauke_7T.json"
FITTER = (
    WORKSPACE_ROOT
    / "forD/forD/classical_fitting/Problem_regularized_standalone.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit all 7-T WALINET outputs sequentially and store each result "
            "beside the subject's OriginalData directory."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--expected-subjects",
        type=int,
        default=25,
        help="Abort before fitting unless exactly this many inputs are found.",
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help="Run subjects again even when trained_metadata.json already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the discovered jobs without starting the fitter.",
    )
    return parser.parse_args()


def discover_subjects(data_root: Path) -> list[tuple[Path, Path, Path]]:
    jobs = []
    for data_path in sorted(data_root.glob("*/OriginalData/data_after_walinet.npy")):
        subject_dir = data_path.parent.parent
        mask_path = subject_dir / "masks/brain_mask.npy"
        output_dir = subject_dir / "MetabMapsAfterWalinet"
        if not mask_path.is_file():
            raise FileNotFoundError(
                f"Brain mask missing for {subject_dir.name}: {mask_path}"
            )
        jobs.append((data_path, mask_path, output_dir))
    return jobs


def write_subject_config(
    template: dict, data_path: Path, mask_path: Path, output_dir: Path
) -> Path:
    config = dict(template)
    config["io_config"] = dict(template["io_config"])
    config["io_config"].update(
        {
            "data_path": str(data_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "logging_path": str(output_dir.resolve()),
            "saving_path": str(output_dir.resolve()),
        }
    )
    config["optimizer_config"] = dict(template["optimizer_config"])
    config["optimizer_config"]["max_iter"] = 200

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "fitting_config_used.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")
    return config_path


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    template_path = args.config.resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not template_path.is_file():
        raise FileNotFoundError(f"Fitting config does not exist: {template_path}")
    if not FITTER.is_file():
        raise FileNotFoundError(f"Fitting entry point does not exist: {FITTER}")

    jobs = discover_subjects(data_root)
    if len(jobs) != args.expected_subjects:
        raise RuntimeError(
            f"Found {len(jobs)} subjects, expected {args.expected_subjects}. "
            "Use --expected-subjects to explicitly accept another count."
        )

    with template_path.open("r", encoding="utf-8") as handle:
        template = json.load(handle)

    print(f"Found {len(jobs)} subjects below {data_root}", flush=True)
    failures = []
    for number, (data_path, mask_path, output_dir) in enumerate(jobs, start=1):
        subject_name = data_path.parent.parent.name
        completion_marker = output_dir / "trained_metadata.json"
        if completion_marker.is_file() and not args.rerun_completed:
            print(
                f"[{number:02d}/{len(jobs):02d}] SKIP {subject_name}: already completed",
                flush=True,
            )
            continue

        print(
            f"[{number:02d}/{len(jobs):02d}] START {subject_name}\n"
            f"  data:   {data_path}\n"
            f"  mask:   {mask_path}\n"
            f"  output: {output_dir}",
            flush=True,
        )
        if args.dry_run:
            continue

        subject_config = write_subject_config(
            template, data_path, mask_path, output_dir
        )
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(WORKSPACE_ROOT / "forD"), existing_pythonpath)
            if value
        )
        result = subprocess.run(
            [sys.executable, str(FITTER), "--config", str(subject_config)],
            cwd=WORKSPACE_ROOT / "forD",
            env=environment,
            check=False,
        )
        if result.returncode == 0:
            print(
                f"[{number:02d}/{len(jobs):02d}] DONE {subject_name}", flush=True
            )
        else:
            failures.append((subject_name, result.returncode))
            print(
                f"[{number:02d}/{len(jobs):02d}] FAILED {subject_name} "
                f"(exit code {result.returncode}); continuing",
                flush=True,
            )

    if args.dry_run:
        print("Dry run complete; no fitting was started.", flush=True)
        return 0
    if failures:
        summary = ", ".join(f"{name} ({code})" for name, code in failures)
        print(f"Completed with {len(failures)} failure(s): {summary}", flush=True)
        return 1
    print("All subject fits completed successfully.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
