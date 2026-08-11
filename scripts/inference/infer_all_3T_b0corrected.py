#!/usr/bin/env python3
"""Run WALINET on every 3T B0-corrected ``OriginalData/data.npy``."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.inference.fid_inference import infer_fid  # noqa: E402


DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "3T" / "B0corrected_wo_LipidMask"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "3T_Final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer every <scanner>/<subject>/<resolution>/OriginalData/data.npy "
            "below the 3T B0-corrected data branch."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", default="model_best.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing data_after_walinet.npy after validating the new output.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List inputs, masks and outputs without running inference.",
    )
    return parser.parse_args()


def find_inputs(data_root: Path) -> list[Path]:
    # This deliberately enforces exactly the documented hierarchy instead of
    # accepting arbitrary recursively located data.npy files.
    return sorted(data_root.glob("*/*/*/OriginalData/data.npy"))


def validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    data_root = args.data_root.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()

    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_dir}")
    checkpoint_path = model_dir / args.checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    return data_root, model_dir


def main() -> None:
    args = parse_args()
    data_root, model_dir = validate_paths(args)
    inputs = find_inputs(data_root)

    if not inputs:
        raise FileNotFoundError(f"No OriginalData/data.npy found below: {data_root}")

    print(f"Data root:  {data_root}", flush=True)
    print(f"Model:      {model_dir}", flush=True)
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Device:     {args.device}", flush=True)
    print(f"Batch size: {args.batch_size}", flush=True)
    print(f"Datasets:   {len(inputs)}", flush=True)
    print(f"Overwrite:  {args.overwrite}", flush=True)
    print(flush=True)

    completed = 0
    skipped = 0

    for index, data_path in enumerate(inputs, start=1):
        resolution_dir = data_path.parent.parent
        mask_path = resolution_dir / "masks" / "brain_mask.npy"
        output_path = data_path.with_name("data_after_walinet.npy")
        temporary_path = data_path.with_name(
            f".data_after_walinet.{os.getpid()}.tmp.npy"
        )

        print(f"[{index:03d}/{len(inputs):03d}] Data:   {data_path}", flush=True)
        print(f"          Mask:   {mask_path}", flush=True)
        print(f"          Output: {output_path}", flush=True)

        if not mask_path.is_file():
            raise FileNotFoundError(f"Matching brain mask is missing: {mask_path}")

        if output_path.exists() and not args.overwrite:
            skipped += 1
            print("          SKIP: output already exists (use --overwrite).", flush=True)
            continue

        if args.dry_run:
            print("          DRY RUN", flush=True)
            continue

        if temporary_path.exists():
            raise FileExistsError(f"Temporary output already exists: {temporary_path}")

        source = np.load(data_path, mmap_mode="r", allow_pickle=False)
        expected_shape = source.shape
        del source

        try:
            infer_fid(
                fid=data_path,
                model_dir=model_dir,
                output_path=temporary_path,
                fid_axis=-1,
                headmask=mask_path,
                checkpoint=args.checkpoint,
                batch_size=args.batch_size,
                device=args.device,
            )

            candidate = np.load(temporary_path, mmap_mode="r", allow_pickle=False)
            if candidate.shape != expected_shape:
                raise RuntimeError(
                    f"Output shape {candidate.shape} does not match input "
                    f"shape {expected_shape}: {data_path}"
                )
            if candidate.dtype != np.complex64:
                raise RuntimeError(
                    f"Expected complex64 output, got {candidate.dtype}: {temporary_path}"
                )
            if not np.isfinite(candidate).all():
                raise RuntimeError(f"Output contains non-finite values: {temporary_path}")
            del candidate

            # Atomic within the same directory: an old output remains intact
            # until the complete new file has passed validation.
            temporary_path.replace(output_path)
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise

        completed += 1
        print("          DONE", flush=True)

    print(flush=True)
    if args.dry_run:
        print(f"Dry run finished; inspected {len(inputs)} dataset(s).", flush=True)
    else:
        print(
            f"Finished: {completed} inferred, {skipped} skipped, "
            f"{len(inputs)} discovered.",
            flush=True,
        )


if __name__ == "__main__":
    main()
