"""Python interface to the Julia/MRSI.jl B0 correction."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Union

import numpy as np


PathLike = Union[str, Path]


def correct_b0(
    combined_csi_path: PathLike,
    *,
    output_dir: PathLike | None = None,
    dat_path: PathLike | None = None,
    julia_executable: PathLike = "julia",
    julia_project: PathLike | None = None,
) -> np.ndarray | tuple[Path, Path, Path]:
    """Perform B0 correction of a CombinedCSI.mat file using MRSI.jl.

    The CombinedCSI.mat file must contain:

        csi.Data
            Complex FID data.

        mask
            Spatial brain mask.

        Par.Paths.csi_path
            Path to the corresponding Siemens .dat file, unless an
            explicit dat_path override is supplied.

    Parameters
    ----------
    combined_csi_path
        Path to CombinedCSI.mat.

    output_dir
        If None, Julia writes its results to a temporary local directory.
        The corrected FIDs are loaded and returned as a NumPy array, and
        the temporary files are deleted automatically.

        If supplied, Julia writes its results directly to this directory.
        In this case the result is not loaded into memory. Instead, paths
        to data_B0corrected.npy, B0_estimation.npy, and brain_mask.npy
        are returned.

    dat_path
        Optional path to the Siemens .dat file. If supplied, this
        overrides Par.Paths.csi_path from CombinedCSI.mat.

    julia_executable
        Julia executable. Defaults to "julia".

    julia_project
        Optional Julia project/environment containing MRSI.jl and the
        other required Julia dependencies.

    Returns
    -------
    np.ndarray
        Corrected FIDs if output_dir is None.

    tuple[Path, Path, Path]
        Paths to data_B0corrected.npy, B0_estimation.npy, and
        brain_mask.npy if output_dir is supplied.
    """

    combined_csi_path = (
        Path(combined_csi_path)
        .expanduser()
        .resolve()
    )

    if not combined_csi_path.is_file():
        raise FileNotFoundError(
            f"CombinedCSI.mat does not exist: {combined_csi_path}"
        )

    if combined_csi_path.suffix.lower() != ".mat":
        raise ValueError(
            "combined_csi_path must point to a .mat file."
        )

    if dat_path is not None:
        dat_path = (
            Path(dat_path)
            .expanduser()
            .resolve()
        )

        if not dat_path.is_file():
            raise FileNotFoundError(
                f"Siemens .dat file does not exist: {dat_path}"
            )

    julia_script = (
        Path(__file__)
        .with_suffix(".jl")
        .resolve()
    )

    if not julia_script.is_file():
        raise FileNotFoundError(
            f"Julia B0 correction script does not exist: {julia_script}"
        )

    julia_executable = str(julia_executable)

    if shutil.which(julia_executable) is None:
        executable_path = Path(
            julia_executable
        ).expanduser()

        if not executable_path.is_file():
            raise FileNotFoundError(
                f"Julia executable not found: {julia_executable}"
            )

        julia_executable = str(
            executable_path.resolve()
        )

    # ------------------------------------------------------------
    # In-memory mode
    # ------------------------------------------------------------
    if output_dir is None:
        with tempfile.TemporaryDirectory(
            prefix="walinet_b0_"
        ) as temporary_dir:
            temporary_dir = Path(
                temporary_dir
            )

            _run_julia_b0_correction(
                combined_csi_path=combined_csi_path,
                output_dir=temporary_dir,
                julia_script=julia_script,
                julia_executable=julia_executable,
                julia_project=julia_project,
                dat_path=dat_path,
            )

            corrected_path = (
                temporary_dir
                / "data_B0corrected.npy"
            )

            b0_path = (
                temporary_dir
                / "B0_estimation.npy"
            )

            mask_path = (
                temporary_dir
                / "brain_mask.npy"
            )

            _check_outputs(
                corrected_path,
                b0_path,
                mask_path,
            )

            print(
                "[correct_b0] Loading corrected FIDs into memory."
            )

            corrected_fids = np.load(
                corrected_path,
                allow_pickle=False,
            )

            # TemporaryDirectory deletes all temporary files after
            # leaving this block.
            return corrected_fids

    # ------------------------------------------------------------
    # File output mode
    # ------------------------------------------------------------
    output_dir = (
        Path(output_dir)
        .expanduser()
        .resolve()
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    _run_julia_b0_correction(
        combined_csi_path=combined_csi_path,
        output_dir=output_dir,
        julia_script=julia_script,
        julia_executable=julia_executable,
        julia_project=julia_project,
        dat_path=dat_path,
    )

    corrected_path = (
        output_dir
        / "data_B0corrected.npy"
    )

    b0_path = (
        output_dir
        / "B0_estimation.npy"
    )

    mask_path = (
        output_dir
        / "brain_mask.npy"
    )

    _check_outputs(
        corrected_path,
        b0_path,
        mask_path,
    )

    return corrected_path, b0_path, mask_path


def _run_julia_b0_correction(
    *,
    combined_csi_path: Path,
    output_dir: Path,
    julia_script: Path,
    julia_executable: str,
    julia_project: PathLike | None,
    dat_path: Path | None,
) -> None:
    command = [
        julia_executable,
    ]

    if julia_project is not None:
        julia_project = (
            Path(julia_project)
            .expanduser()
            .resolve()
        )

        if not julia_project.is_dir():
            raise FileNotFoundError(
                f"Julia project directory does not exist: {julia_project}"
            )

        command.append(
            f"--project={julia_project}"
        )

    command.extend(
        [
            str(julia_script),
            str(combined_csi_path),
            str(output_dir),
        ]
    )

    # Optional override for Par.Paths.csi_path.
    if dat_path is not None:
        command.append(
            str(dat_path)
        )

    print(
        "[correct_b0] Starting Julia B0 correction..."
    )

    if dat_path is not None:
        print(
            f"[correct_b0] Using Siemens .dat override: {dat_path}"
        )

    subprocess.run(
        command,
        check=True,
    )

    print(
        "[correct_b0] Julia B0 correction finished."
    )


def _check_outputs(
    corrected_path: Path,
    b0_path: Path,
    mask_path: Path,
) -> None:
    missing = [
        path
        for path in (
            corrected_path,
            b0_path,
            mask_path,
        )
        if not path.is_file()
    ]

    if missing:
        paths = "\n".join(
            f"  {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Julia B0 correction finished without "
            "creating all expected files:\n"
            f"{paths}"
        )