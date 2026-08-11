#!/usr/bin/env python3
"""Export an acquisition-matched WALINET LCModel basis as portable NPZ."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from walinet.training_data.lcmodel_basis.acquisition import (  # noqa: E402
    prepare_basis_for_acquisition,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bandwidth-hz", type=float, required=True)
    parser.add_argument("--n-timepoints", type=int, required=True)
    parser.add_argument("--dataset-name", default="clean_fid")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = args.library.expanduser().resolve()
    output = args.output.expanduser().resolve()

    prepared = prepare_basis_for_acquisition(
        library,
        target_bandwidth=args.bandwidth_hz,
        target_n_timepoints=args.n_timepoints,
        component_names=None,
        dataset_name=args.dataset_name,
        output_dtype=np.complex64,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        basis_fids=np.ascontiguousarray(prepared.fids, dtype=np.complex64),
        basis_names=np.asarray(prepared.names, dtype=np.str_),
        dwell_time_seconds=np.asarray(prepared.dwell_time, dtype=np.float64),
        bandwidth_hz=np.asarray(prepared.bandwidth, dtype=np.float64),
        requested_bandwidth_hz=np.asarray(
            prepared.requested_bandwidth, dtype=np.float64
        ),
        n_timepoints=np.asarray(prepared.n_timepoints, dtype=np.int64),
        hz_per_ppm=np.asarray(prepared.hz_per_ppm, dtype=np.float64),
        ppm_reference=np.asarray(prepared.ppm_reference, dtype=np.float64),
        source_bandwidth_hz=np.asarray(prepared.source_bandwidth, dtype=np.float64),
        source_dwell_time_seconds=np.asarray(
            prepared.source_dwell_time, dtype=np.float64
        ),
        source_n_points=np.asarray(prepared.source_n_points, dtype=np.int64),
        dataset_name=np.asarray(prepared.dataset_name, dtype=np.str_),
        source_library=np.asarray(str(prepared.library_path), dtype=np.str_),
    )

    # Independent disk round-trip validation without pickle support.
    with np.load(output, allow_pickle=False) as saved:
        fids = saved["basis_fids"]
        names = saved["basis_names"]
        if fids.shape != (prepared.n_metabolites, prepared.n_timepoints):
            raise RuntimeError(f"Unexpected saved basis shape: {fids.shape}")
        if fids.dtype != np.complex64 or not fids.flags.c_contiguous:
            raise RuntimeError(
                f"Saved basis must be C-contiguous complex64, got {fids.dtype}."
            )
        if names.tolist() != prepared.names:
            raise RuntimeError("Saved basis names/order failed validation.")
        if not np.isfinite(fids).all():
            raise RuntimeError("Saved basis contains non-finite values.")

    print(f"Saved and validated portable basis: {output}")
    print(f"Shape: {prepared.fids.shape}, dtype: {prepared.fids.dtype}")
    print("Components: " + ", ".join(prepared.names))


if __name__ == "__main__":
    main()
