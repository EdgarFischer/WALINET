"""WALINET-to-forD classical fitting pipeline.

Supported inputs
----------------
1. NumPy input:
       data.npy + separate mask.npy

2. CombinedCSI.mat:
       csi.Data and mask are used directly.

For CombinedCSI.mat with B0 correction enabled, Python does not read the
MAT file. Julia/MAT.jl loads csi.Data and mask, performs B0 correction,
and writes a consistent NumPy data/mask pair to /dev/shm.

WALINET is always applied.

Before forD fitting, the WALINET-cleaned FIDs and mask are provided as
temporary .npy files in /dev/shm. This keeps the existing file-based forD
interface unchanged while avoiding temporary Ceph I/O.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import torch

from walinet.data.combined_csi_io import load_combined_csi
from walinet.inference.fid_inference import infer_fid
from walinet.preprocessing.b0_correction import correct_b0


PathLike = Union[str, Path]


def run_walinet_ford_pipeline(
    data_path: PathLike,
    mask_path: PathLike | None,
    walinet_model_dir: PathLike,
    ford_config_template: PathLike,
    output_path: PathLike,
    gpu_number: int,
    *,
    fid_axis: Union[int, str] = "auto",
    walinet_checkpoint: str = "model_best.pt",
    walinet_batch_size: int = 200,
    b0_correction: bool = False,
    dat_path: PathLike | None = None,
    julia_executable: PathLike = "julia",
    julia_project: PathLike | None = None,
    shm_dir: PathLike = "/dev/shm",
) -> Path:
    """Run WALINET inference followed by classical forD fitting.

    Input
    -----
    data_path may point to either:

    - a .npy FID array, in which case mask_path is required, or
    - a CombinedCSI.mat file.

    For CombinedCSI.mat with B0 correction enabled, Python does not read
    the MAT file. Julia/MAT.jl loads both csi.Data and mask, performs B0
    correction, and writes data_B0corrected.npy and brain_mask.npy to
    shm_dir.

    WALINET always operates on NumPy input after preprocessing.

    The WALINET-cleaned FIDs and mask are provided to forD through
    temporary .npy files in shm_dir.
    """

    data_path = _existing_input_file(
        data_path,
        "data_path",
    )

    model_dir = _existing_directory(
        walinet_model_dir,
        "walinet_model_dir",
    )

    config_path = _existing_file(
        ford_config_template,
        "ford_config_template",
    )

    shm_root = _existing_directory(
        shm_dir,
        "shm_dir",
    )

    output_dir = (
        Path(output_path)
        .expanduser()
        .resolve()
    )

    # ------------------------------------------------------------
    # GPU
    # ------------------------------------------------------------
    if not isinstance(gpu_number, int) or isinstance(gpu_number, bool):
        raise TypeError(
            "gpu_number must be an integer."
        )

    if gpu_number < 0:
        raise ValueError(
            "gpu_number must be >= 0."
        )

    device = torch.device(
        f"cuda:{gpu_number}"
    )

    # ------------------------------------------------------------
    # Output
    # ------------------------------------------------------------
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------
    # Load forD config
    # ------------------------------------------------------------
    with config_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        effective_config = json.load(
            handle
        )

    basis_path = _resolve_config_file(
        effective_config["io_config"].get(
            "basis_path"
        ),
        config_path,
        "io_config.basis_path",
    )

    # ------------------------------------------------------------
    # One RAM-backed temporary directory for the complete pipeline
    # ------------------------------------------------------------
    with tempfile.TemporaryDirectory(
        prefix="walinet_ford_",
        dir=shm_root,
    ) as temporary_directory:
        temporary_directory = Path(
            temporary_directory
        )

        # --------------------------------------------------------
        # Prepare input
        # --------------------------------------------------------
        (
            walinet_input_path,
            pipeline_mask_path,
            resolved_fid_axis,
            input_shape,
            mask_shape,
        ) = _load_pipeline_input(
            data_path=data_path,
            mask_path=mask_path,
            fid_axis=fid_axis,
            b0_correction=b0_correction,
            dat_path=dat_path,
            julia_executable=julia_executable,
            julia_project=julia_project,
            temporary_directory=temporary_directory,
        )

        print(
            f"[pipeline] Input data shape: {input_shape}",
            flush=True,
        )

        print(
            f"[pipeline] Mask shape: {mask_shape}",
            flush=True,
        )

        print(
            f"[pipeline] FID axis: {resolved_fid_axis}",
            flush=True,
        )

        # --------------------------------------------------------
        # WALINET
        # --------------------------------------------------------
        print(
            f"[pipeline] Starting WALINET inference on {device} "
            f"with model {model_dir}",
            flush=True,
        )

        # WALINET reads the prepared NumPy data and mask.
        #
        # For B0-corrected CombinedCSI input these are exactly the
        # .npy files written by Julia into /dev/shm.
        cleaned_fid = infer_fid(
            walinet_input_path,
            model_dir,
            headmask=pipeline_mask_path,
            fid_axis=resolved_fid_axis,
            checkpoint=walinet_checkpoint,
            batch_size=walinet_batch_size,
            device=device,
            output_path=None,
        )

        # forD expects the spectral/FID dimension last.
        cleaned_fid = np.moveaxis(
            cleaned_fid,
            resolved_fid_axis,
            -1,
        )

        cleaned_fid = np.asarray(
            cleaned_fid,
            dtype=np.complex64,
        )

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(
            f"[pipeline] WALINET inference finished; "
            f"cleaned shape: {cleaned_fid.shape}",
            flush=True,
        )

        # --------------------------------------------------------
        # Import forD lazily
        # --------------------------------------------------------
        # WALINET-only use therefore does not require forD dependencies.
        from forD.classical_fitting.Problem_regularized_standalone import Problem
        from forD.classical_fitting.config import Configuration

        # --------------------------------------------------------
        # Temporary RAM-backed forD input
        # --------------------------------------------------------
        ford_data_path = (
            temporary_directory
            / "data_WALINET.npy"
        )

        # forD's diagnostics look for ``magnitude.npy`` beside the configured
        # data_path. Since the fitted WALINET data live temporarily in RAM,
        # mirror the anatomical magnitude from beside the original input into
        # that same temporary directory. This affects diagnostic plots only.
        source_magnitude_path = data_path.parent / "magnitude.npy"
        ford_magnitude_path = temporary_directory / "magnitude.npy"

        if source_magnitude_path.is_file():
            # Fail early on a corrupt/non-NumPy file instead of copying it and
            # producing a less informative error at the end of the fit.
            magnitude = np.load(
                source_magnitude_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            magnitude_shape = tuple(magnitude.shape)
            del magnitude

            shutil.copyfile(
                source_magnitude_path,
                ford_magnitude_path,
            )

            print(
                f"[pipeline] Mirrored diagnostic magnitude to RAM: "
                f"{source_magnitude_path} -> {ford_magnitude_path} "
                f"(shape {magnitude_shape})",
                flush=True,
            )
        else:
            print(
                f"[pipeline] No diagnostic magnitude found beside input: "
                f"{source_magnitude_path}",
                flush=True,
            )

        # pipeline_mask_path is already a .npy file in /dev/shm.
        ford_mask_path = pipeline_mask_path

        print(
            "[pipeline] Writing temporary forD input to RAM:",
            flush=True,
        )

        print(
            f"[pipeline]   data: {ford_data_path}",
            flush=True,
        )

        print(
            f"[pipeline]   mask: {ford_mask_path}",
            flush=True,
        )

        np.save(
            ford_data_path,
            cleaned_fid,
        )

        # The mask is already stored in /dev/shm by the input
        # preparation step, so it is deliberately NOT rewritten here.

        del cleaned_fid

        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

        # --------------------------------------------------------
        # Configure forD
        # --------------------------------------------------------
        effective_config["io_config"].update(
            {
                "basis_path": str(
                    basis_path
                ),
                "data_path": str(
                    ford_data_path
                ),
                "mask_path": str(
                    ford_mask_path
                ),
                "logging_path": str(
                    output_dir
                ),
                "saving_path": str(
                    output_dir
                ),
            }
        )

        effective_config["pytorch_config"]["device"] = str(
            device
        )

        # Save the exact configuration used by forD.
        #
        # data_path and mask_path refer to temporary /dev/shm files,
        # which intentionally disappear after the pipeline finishes.
        used_config_path = (
            output_dir
            / "fitting_config_used.json"
        )

        with used_config_path.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                effective_config,
                handle,
                indent=2,
            )

            handle.write(
                "\n"
            )

        config = Configuration.from_dict(
            effective_config
        )

        # --------------------------------------------------------
        # PyTorch / forD runtime configuration
        # --------------------------------------------------------
        torch.set_default_device(
            device
        )

        torch.set_default_dtype(
            config.pytorch_config.default_type
            or torch.float32
        )

        if (
            config.pytorch_config.float32_matmul_precision
            is not None
        ):
            torch.set_float32_matmul_precision(
                config.pytorch_config.float32_matmul_precision
            )

        if (
            config.pytorch_config.num_threads
            is not None
        ):
            torch.set_num_threads(
                config.pytorch_config.num_threads
            )

        torch.manual_seed(
            0
        )

        # --------------------------------------------------------
        # forD fitting
        # --------------------------------------------------------
        print(
            f"[pipeline] Starting classical forD fitting; "
            f"output: {output_dir}",
            flush=True,
        )

        # forD reads data_path and mask_path directly from its config.
        problem = Problem(
            config
        )

        problem._optimize()

        print(
            "[pipeline] forD fitting finished.",
            flush=True,
        )

    # TemporaryDirectory automatically removes all /dev/shm files here.
    print(
        "[pipeline] Temporary /dev/shm inputs removed.",
        flush=True,
    )

    print(
        "[pipeline] WALINET + forD pipeline completed successfully.",
        flush=True,
    )

    return output_dir


def _load_pipeline_input(
    *,
    data_path: Path,
    mask_path: PathLike | None,
    fid_axis: Union[int, str],
    b0_correction: bool,
    dat_path: PathLike | None,
    julia_executable: PathLike,
    julia_project: PathLike | None,
    temporary_directory: Path,
) -> tuple[
    Path,
    Path,
    int,
    tuple[int, ...],
    tuple[int, ...],
]:
    """Prepare WALINET input as a consistent NumPy data/mask pair.

    For CombinedCSI.mat with B0 correction enabled, Python does not
    read the MAT file. Julia loads both csi.Data and mask, performs the
    correction, and writes both arrays directly to temporary_directory.
    """

    suffix = data_path.suffix.lower()

    # ------------------------------------------------------------
    # NumPy input
    # ------------------------------------------------------------
    if suffix == ".npy":
        if b0_correction:
            raise ValueError(
                "B0 correction requires CombinedCSI.mat input because "
                "the required acquisition metadata are stored in the MAT file."
            )

        if dat_path is not None:
            raise ValueError(
                "dat_path is only meaningful when B0 correction is enabled "
                "for CombinedCSI.mat input."
            )

        if mask_path is None:
            raise ValueError(
                "mask_path is required when data_path points to a .npy file."
            )

        mask_path = _existing_npy(
            mask_path,
            "mask_path",
        )

        print(
            f"[pipeline] Using NumPy data: {data_path}",
            flush=True,
        )

        print(
            f"[pipeline] Loading NumPy mask: {mask_path}",
            flush=True,
        )

        mask = np.asarray(
            np.load(
                mask_path,
                allow_pickle=False,
            ),
            dtype=bool,
        )

        # Keep the mask used by WALINET and forD in the common
        # /dev/shm temporary directory.
        pipeline_mask_path = (
            temporary_directory
            / "brain_mask.npy"
        )

        np.save(
            pipeline_mask_path,
            mask,
        )

        walinet_input_path = data_path

        del mask

    # ------------------------------------------------------------
    # CombinedCSI.mat input
    # ------------------------------------------------------------
    elif suffix == ".mat":
        if mask_path is not None:
            raise ValueError(
                "mask_path must be None for CombinedCSI.mat input. "
                "The embedded CombinedCSI mask is used automatically."
            )

        # --------------------------------------------------------
        # CombinedCSI + B0 correction
        # --------------------------------------------------------
        if b0_correction:
            print(
                "[pipeline] B0 correction enabled.",
                flush=True,
            )

            print(
                "[pipeline] CombinedCSI.mat will be read exclusively "
                "by Julia/MAT.jl.",
                flush=True,
            )

            (
                corrected_path,
                b0_path,
                julia_mask_path,
            ) = correct_b0(
                combined_csi_path=data_path,
                dat_path=dat_path,
                output_dir=temporary_directory,
                julia_executable=julia_executable,
                julia_project=julia_project,
            )

            walinet_input_path = corrected_path
            pipeline_mask_path = julia_mask_path

            print(
                "[pipeline] B0 correction finished.",
                flush=True,
            )

            print(
                f"[pipeline]   corrected data: {walinet_input_path}",
                flush=True,
            )

            print(
                f"[pipeline]   mask: {pipeline_mask_path}",
                flush=True,
            )

            print(
                f"[pipeline]   B0 map: {b0_path}",
                flush=True,
            )

        # --------------------------------------------------------
        # CombinedCSI without B0 correction
        # --------------------------------------------------------
        else:
            if dat_path is not None:
                raise ValueError(
                    "dat_path was supplied, but B0 correction is disabled."
                )

            print(
                f"[pipeline] Loading CombinedCSI.mat without "
                f"B0 correction: {data_path}",
                flush=True,
            )

            data, mask = load_combined_csi(
                data_path
            )

            data = np.asarray(
                data
            )

            mask = np.asarray(
                mask,
                dtype=bool,
            )

            walinet_input_path = (
                temporary_directory
                / "data_input.npy"
            )

            pipeline_mask_path = (
                temporary_directory
                / "brain_mask.npy"
            )

            np.save(
                walinet_input_path,
                data,
            )

            np.save(
                pipeline_mask_path,
                mask,
            )

            del data
            del mask

    else:
        raise ValueError(
            "data_path must point to either a .npy file "
            f"or a CombinedCSI .mat file: {data_path}"
        )

    # ------------------------------------------------------------
    # Validate prepared NumPy files
    # ------------------------------------------------------------
    #
    # mmap_mode lets us inspect shape/axes without loading another full
    # copy of the FID array into memory.
    data_probe = np.load(
        walinet_input_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    mask_probe = np.load(
        pipeline_mask_path,
        mmap_mode="r",
        allow_pickle=False,
    )

    input_shape = tuple(
        data_probe.shape
    )

    mask_shape = tuple(
        mask_probe.shape
    )

    if data_probe.ndim < 2:
        raise ValueError(
            f"FID data must have at least two dimensions; "
            f"got shape {input_shape}."
        )

    resolved_fid_axis = _resolve_fid_axis(
        fid_axis,
        input_shape,
    )

    spatial_shape = (
        input_shape[:resolved_fid_axis]
        + input_shape[resolved_fid_axis + 1 :]
    )

    if spatial_shape != mask_shape:
        raise ValueError(
            f"Data spatial shape {spatial_shape} does not match "
            f"mask shape {mask_shape}."
        )

    del data_probe
    del mask_probe

    return (
        walinet_input_path,
        pipeline_mask_path,
        resolved_fid_axis,
        input_shape,
        mask_shape,
    )


def _existing_input_file(
    path: PathLike,
    name: str,
) -> Path:
    """Require an existing .npy or .mat input file."""
    result = _existing_file(
        path,
        name,
    )

    if result.suffix.lower() not in {
        ".npy",
        ".mat",
    }:
        raise ValueError(
            f"{name} must point to a .npy or .mat file: {result}"
        )

    return result


def _existing_npy(
    path: PathLike,
    name: str,
) -> Path:
    result = _existing_file(
        path,
        name,
    )

    if result.suffix.lower() != ".npy":
        raise ValueError(
            f"{name} must point to a .npy file: {result}"
        )

    return result


def _existing_file(
    path: PathLike,
    name: str,
) -> Path:
    result = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not result.is_file():
        raise FileNotFoundError(
            f"{name} does not exist: {result}"
        )

    return result


def _existing_directory(
    path: PathLike,
    name: str,
) -> Path:
    result = (
        Path(path)
        .expanduser()
        .resolve()
    )

    if not result.is_dir():
        raise FileNotFoundError(
            f"{name} does not exist: {result}"
        )

    return result


def _resolve_config_file(
    configured_path: str | None,
    config_path: Path,
    name: str,
) -> Path:
    if not configured_path:
        raise ValueError(
            f"{name} must be set in {config_path}."
        )

    path = Path(
        configured_path
    ).expanduser()

    if path.is_absolute():
        candidates = [
            path
        ]

    else:
        candidates = [
            Path.cwd() / path,
            config_path.parent / path,
            config_path.parent.parent / path,
        ]

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved.is_file():
            print(
                f"[pipeline] Resolved {name}: {resolved}",
                flush=True,
            )

            return resolved

    checked = "\n  ".join(
        str(
            candidate.resolve()
        )
        for candidate in candidates
    )

    raise FileNotFoundError(
        f"Could not resolve {name}; checked:\n  {checked}"
    )


def _resolve_fid_axis(
    fid_axis: Union[int, str],
    shape: tuple[int, ...],
) -> int:
    """Resolve the FID axis.

    With fid_axis='auto', the longest dimension is interpreted as the
    FID dimension.
    """
    if fid_axis == "auto":
        axis = int(
            np.argmax(
                shape
            )
        )

        print(
            f"[pipeline] Automatically detected FID axis "
            f"{axis} with length {shape[axis]}.",
            flush=True,
        )

        return axis

    if isinstance(
        fid_axis,
        (int, np.integer),
    ):
        axis = int(
            fid_axis
        )

        if axis < 0:
            axis += len(
                shape
            )

        if not 0 <= axis < len(shape):
            raise np.AxisError(
                fid_axis,
                ndim=len(shape),
            )

        return axis

    raise TypeError(
        "fid_axis must be 'auto' or an integer."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WALINET inference followed by classical forD fitting. "
            "Input may be data.npy + mask.npy or CombinedCSI.mat."
        )
    )

    parser.add_argument(
        "--data",
        required=True,
        type=Path,
        help=(
            "Input data.npy or CombinedCSI.mat."
        ),
    )

    parser.add_argument(
        "--mask",
        type=Path,
        default=None,
        help=(
            "Brain mask .npy. Required for .npy input; "
            "must be omitted for CombinedCSI.mat."
        ),
    )

    parser.add_argument(
        "--walinet-model",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--ford-config",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--gpu",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--fid-axis",
        default="auto",
    )

    parser.add_argument(
        "--walinet-checkpoint",
        default="model_best.pt",
    )

    parser.add_argument(
        "--walinet-batch-size",
        default=200,
        type=int,
    )

    parser.add_argument(
        "--b0-correction",
        action="store_true",
        help=(
            "Apply Julia/MRSI.jl B0 correction before WALINET. "
            "Only supported for CombinedCSI.mat input."
        ),
    )

    parser.add_argument(
        "--dat-path",
        type=Path,
        default=None,
        help=(
            "Optional Siemens .dat override for B0 correction. "
            "If omitted, Par.Paths.csi_path from CombinedCSI.mat is used."
        ),
    )

    parser.add_argument(
        "--julia-executable",
        default="julia",
        help=(
            "Julia executable or full path to the Julia binary."
        ),
    )

    parser.add_argument(
        "--julia-project",
        type=Path,
        default=None,
        help=(
            "Julia project containing MRSI.jl and its dependencies."
        ),
    )

    parser.add_argument(
        "--shm-dir",
        type=Path,
        default=Path("/dev/shm"),
        help=(
            "RAM-backed temporary directory used for the pipeline. "
            "Default: /dev/shm"
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    fid_axis = args.fid_axis

    if fid_axis != "auto":
        try:
            fid_axis = int(
                fid_axis
            )

        except ValueError as error:
            raise ValueError(
                "--fid-axis must be 'auto' or an integer."
            ) from error

    run_walinet_ford_pipeline(
        data_path=args.data,
        mask_path=args.mask,
        walinet_model_dir=args.walinet_model,
        ford_config_template=args.ford_config,
        output_path=args.output,
        gpu_number=args.gpu,
        fid_axis=fid_axis,
        walinet_checkpoint=args.walinet_checkpoint,
        walinet_batch_size=args.walinet_batch_size,
        b0_correction=args.b0_correction,
        dat_path=args.dat_path,
        julia_executable=args.julia_executable,
        julia_project=args.julia_project,
        shm_dir=args.shm_dir,
    )


if __name__ == "__main__":
    main()
