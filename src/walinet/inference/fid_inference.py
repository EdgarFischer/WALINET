from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Union

import numpy as np
import torch
import yaml

from walinet.data.combined_csi_io import (
    load_combined_csi,
    save_combined_csi,
)
from walinet.inference.inference import (
    _load_checkpoint_state_dict,
    _load_model_and_params,
    _resolve_normalization,
)
from walinet.preprocessing.b0_correction import correct_b0


PathLike = Union[str, Path]


@dataclass(frozen=True)
class AcquisitionInfo:
    n_timepoints: int
    min_n_timepoints: int
    max_n_timepoints: int
    zero_filling: bool

    def __post_init__(self) -> None:
        if not (
            0 < self.min_n_timepoints
            <= self.max_n_timepoints
            <= self.n_timepoints
        ):
            raise ValueError(
                "Expected 0 < minimum <= maximum <= n_timepoints, got "
                f"{self.min_n_timepoints}, {self.max_n_timepoints}, "
                f"{self.n_timepoints}."
            )


def _load_acquisition_info(
    model_dir: Path,
) -> AcquisitionInfo | None:
    """Read acquisition lengths, or return None for older models."""
    configs_dir = model_dir / "configs"
    candidates = []

    for path in sorted(configs_dir.rglob("*.yaml")) + sorted(
        configs_dir.rglob("*.yml")
    ):
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        acquisition = (
            config.get("acquisition")
            if isinstance(config, dict)
            else None
        )

        if not isinstance(acquisition, dict):
            continue

        required = {
            "n_timepoints",
            "min_acquired_n_timepoints",
            "max_acquired_n_timepoints",
        }

        if required.issubset(acquisition):
            candidates.append((path, acquisition))

    if not candidates:
        print(
            "[infer_fid] No saved minimum and maximum FID lengths found; "
            "keeping the input FID length unchanged."
        )
        return None

    if len(candidates) > 1:
        paths = "\n".join(
            f"  {path}"
            for path, _ in candidates
        )
        raise ValueError(
            "Multiple saved simulation configs contain acquisition lengths:\n"
            f"{paths}"
        )

    _, acquisition = candidates[0]

    return AcquisitionInfo(
        n_timepoints=int(acquisition["n_timepoints"]),
        min_n_timepoints=int(
            acquisition["min_acquired_n_timepoints"]
        ),
        max_n_timepoints=int(
            acquisition["max_acquired_n_timepoints"]
        ),
        # This is also the default used by the simulation config loader.
        zero_filling=bool(
            acquisition.get("zero_filling", True)
        ),
    )


def _prepare_fid_length(
    fid: np.ndarray,
    acquisition: AcquisitionInfo | None,
) -> np.ndarray:
    """Crop or append zeros according to the model's training setup."""
    if acquisition is None:
        return fid

    input_length = fid.shape[-1]

    if acquisition.zero_filling:
        target_length = acquisition.n_timepoints
    else:
        target_length = min(
            max(
                input_length,
                acquisition.min_n_timepoints,
            ),
            acquisition.max_n_timepoints,
        )

    if input_length > target_length:
        print(
            f"[infer_fid] Cropping FID length from {input_length} "
            f"to {target_length}."
        )
        return fid[..., :target_length]

    if input_length < target_length:
        print(
            f"[infer_fid] Zero-filling FID length from {input_length} "
            f"to {target_length}."
        )

        padding = [(0, 0)] * fid.ndim
        padding[-1] = (
            0,
            target_length - input_length,
        )

        return np.pad(
            fid,
            padding,
            mode="constant",
        )

    return fid


def _fid_to_spectrum(
    fid: np.ndarray,
) -> np.ndarray:
    return np.fft.fftshift(
        np.fft.fft(
            fid,
            axis=-1,
        ),
        axes=-1,
    )


def _spectrum_to_fid(
    spectrum: np.ndarray,
) -> np.ndarray:
    return np.fft.ifft(
        np.fft.ifftshift(
            spectrum,
            axes=-1,
        ),
        axis=-1,
    )


def _infer_spectra(
    spectra: np.ndarray,
    *,
    model: torch.nn.Module,
    device: torch.device,
    batch_size: int,
    headmask: np.ndarray | None,
    eps: float,
) -> np.ndarray:
    """Run max-abs-normalized U-Net inference on FFT-shifted spectra."""
    spatial_shape = spectra.shape[:-1]
    n_timepoints = spectra.shape[-1]
    flat = spectra.reshape(-1, n_timepoints)

    if headmask is None:
        selected = np.arange(flat.shape[0])

    else:
        headmask = np.asarray(headmask)

        if headmask.shape != spatial_shape:
            raise ValueError(
                f"headmask has shape {headmask.shape}, "
                f"expected {spatial_shape}."
            )

        selected = np.flatnonzero(
            headmask.reshape(-1) > 0
        )

    selected_spectra = flat[selected]

    valid = np.isfinite(
        selected_spectra
    ).all(axis=1)

    valid_indices = selected[valid]
    selected_spectra = selected_spectra[valid]

    clean = np.zeros(
        flat.shape,
        dtype=np.complex64,
    )

    model.to(device).eval()

    with torch.no_grad():
        for start in range(
            0,
            len(selected_spectra),
            batch_size,
        ):
            batch_np = selected_spectra[
                start : start + batch_size
            ]

            batch = torch.as_tensor(
                batch_np,
                dtype=torch.cfloat,
                device=device,
            )

            norm = torch.amax(
                torch.abs(batch),
                dim=1,
                keepdim=True,
            )

            norm = torch.clamp(
                norm,
                min=eps,
            )

            normalized = batch / norm

            network_input = torch.stack(
                (
                    normalized.real,
                    normalized.imag,
                ),
                dim=1,
            )

            output = model(
                network_input
            )[:, :2, :]

            nuisance = torch.complex(
                output[:, 0, :],
                output[:, 1, :],
            ) * norm

            clean_batch = batch - nuisance

            clean[
                valid_indices[
                    start : start + batch_size
                ]
            ] = (
                clean_batch
                .cpu()
                .numpy()
                .astype(np.complex64)
            )

    return clean.reshape(
        *spatial_shape,
        n_timepoints,
    )


def infer_fid(
    fid: Union[np.ndarray, str, Path],
    model_dir: Union[str, Path],
    *,
    output_path: Union[str, Path, None] = None,
    fid_axis: Union[int, str] = "auto",
    headmask: Union[np.ndarray, str, Path, None] = None,
    checkpoint: str = "model_best.pt",
    batch_size: int = 200,
    device: Union[str, torch.device, None] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Remove nuisance signals from complex FIDs with a trained U-Net.

    ``fid`` and ``headmask`` may be NumPy arrays or paths to ``.npy`` files.

    If ``fid_axis="auto"``, the longest input axis is interpreted as the
    FID axis.
    """
    if isinstance(fid, (str, Path)):
        fid_path = Path(
            fid
        ).expanduser()

        if fid_path.suffix.lower() != ".npy":
            raise ValueError(
                "Only .npy FID input is currently supported by infer_fid(). "
                "Use infer_combined_csi() for CombinedCSI.mat files."
            )

        fid = np.load(
            fid_path,
            allow_pickle=False,
        )

    if not isinstance(fid, np.ndarray):
        raise TypeError(
            "fid must be a numpy.ndarray or a path to a .npy file."
        )

    if fid.ndim == 0:
        raise ValueError(
            "fid must have at least one dimension."
        )

    if not np.issubdtype(
        fid.dtype,
        np.number,
    ):
        raise TypeError(
            "fid must contain numeric values."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be > 0."
        )

    if isinstance(headmask, (str, Path)):
        headmask_path = Path(
            headmask
        ).expanduser()

        if headmask_path.suffix.lower() != ".npy":
            raise ValueError(
                "Only .npy headmask input is currently supported."
            )

        headmask = np.load(
            headmask_path,
            allow_pickle=False,
        )

    if (
        headmask is not None
        and not isinstance(headmask, np.ndarray)
    ):
        raise TypeError(
            "headmask must be a numpy.ndarray, "
            "a path to a .npy file, or None."
        )

    if fid_axis == "auto":
        original_axis = int(
            np.argmax(fid.shape)
        )

        print(
            f"[infer_fid] Automatically detected FID axis "
            f"{original_axis} with length {fid.shape[original_axis]}."
        )

    elif isinstance(
        fid_axis,
        (int, np.integer),
    ):
        original_axis = int(fid_axis)

        if original_axis < 0:
            original_axis += fid.ndim

        if not 0 <= original_axis < fid.ndim:
            raise np.AxisError(
                fid_axis,
                ndim=fid.ndim,
            )

    else:
        raise TypeError(
            "fid_axis must be 'auto' or an integer."
        )

    model_dir = Path(
        model_dir
    ).expanduser().resolve()

    model_cls, params, architecture, loaded_dir = (
        _load_model_and_params(
            exp=model_dir.name,
            model_root=model_dir.parent,
            architecture="auto",
        )
    )

    normalization = _resolve_normalization(
        params,
        "auto",
    )

    if (
        architecture != "unet"
        or normalization != "max_abs"
    ):
        raise ValueError(
            "infer_fid currently supports operator-free U-Nets with "
            "max_abs normalization only; found "
            f"architecture={architecture!r}, "
            f"normalization={normalization!r}."
        )

    acquisition = _load_acquisition_info(
        loaded_dir
    )

    prepared = np.moveaxis(
        np.asarray(
            fid,
            dtype=np.complex64,
        ),
        original_axis,
        -1,
    )

    prepared = _prepare_fid_length(
        prepared,
        acquisition,
    )

    spectra = _fid_to_spectrum(
        prepared
    )

    if device is None:
        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    else:
        device = torch.device(
            device
        )

    model = model_cls(
        nLayers=int(
            params["nLayers"]
        ),
        nFilters=int(
            params["nFilters"]
        ),
        dropout=float(
            params.get(
                "dropout",
                0.0,
            )
        ),
        in_channels=int(
            params["in_channels"]
        ),
        out_channels=int(
            params["out_channels"]
        ),
    )

    checkpoint_path = (
        loaded_dir / checkpoint
    )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint does not exist: {checkpoint_path}"
        )

    model.load_state_dict(
        _load_checkpoint_state_dict(
            checkpoint_path,
            device,
        )
    )

    clean_spectra = _infer_spectra(
        spectra,
        model=model,
        device=device,
        batch_size=batch_size,
        headmask=headmask,
        eps=eps,
    )

    clean_fid = _spectrum_to_fid(
        clean_spectra
    )

    clean_fid = np.moveaxis(
        clean_fid,
        -1,
        original_axis,
    )

    clean_fid = np.asarray(
        clean_fid,
        dtype=np.complex64,
    )

    if output_path is not None:
        output_path = Path(
            output_path
        ).expanduser()

        if output_path.suffix.lower() != ".npy":
            raise ValueError(
                "Only .npy output is supported by infer_fid(). "
                "Use infer_combined_csi() to write a CombinedCSI.mat file."
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            output_path,
            clean_fid,
        )

        print(
            f"[infer_fid] Saved cleaned FID: {output_path}"
        )

    return clean_fid


def infer_combined_csi(
    input_path: PathLike,
    model_dir: PathLike,
    output_path: PathLike,
    *,
    fid_axis: Union[int, str] = "auto",
    checkpoint: str = "model_best.pt",
    batch_size: int = 200,
    device: Union[str, torch.device, None] = None,
    eps: float = 1e-8,
    b0_correction: bool = False,
    dat_path: PathLike | None = None,
    julia_executable: PathLike = "julia",
    julia_project: PathLike | None = None,
    shm_dir: PathLike = "/dev/shm",
) -> Path:
    """Run WALINET on csi.Data and save a complete CombinedCSI.mat copy.

    Without B0 correction, the input file is loaded using
    ``load_combined_csi()``. With ``b0_correction=True``, the same Julia/MRSI.jl
    correction used by the WALINET-to-forD pipeline is applied first. WALINET
    then receives ``data_B0corrected.npy`` and ``brain_mask.npy`` directly from
    that correction step.

    The output file always retains all fields from the original CombinedCSI,
    while only ``csi.Data`` is replaced by the B0-corrected and WALINET-cleaned
    FID. ``dat_path`` may override ``Par.Paths.csi_path`` for B0 correction.

    If ``fid_axis="auto"``, the longest axis of ``csi.Data`` is interpreted
    as the FID axis.
    """
    input_path = Path(
        input_path
    ).expanduser().resolve()

    output_path = Path(
        output_path
    ).expanduser().resolve()

    if input_path.suffix.lower() != ".mat":
        raise ValueError(
            f"input_path must point to a .mat file: {input_path}"
        )

    if output_path.suffix.lower() != ".mat":
        raise ValueError(
            f"output_path must end in .mat: {output_path}"
        )

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CombinedCSI.mat does not exist: {input_path}"
        )

    if dat_path is not None and not b0_correction:
        raise ValueError(
            "dat_path is only meaningful when b0_correction=True."
        )

    if b0_correction:
        shm_dir = Path(shm_dir).expanduser().resolve()
        if not shm_dir.is_dir():
            raise FileNotFoundError(
                f"Temporary directory does not exist: {shm_dir}"
            )

        print(
            f"[infer_combined_csi] Applying B0 correction: {input_path}"
        )

        with tempfile.TemporaryDirectory(
            prefix="walinet_combined_csi_",
            dir=shm_dir,
        ) as temporary_directory:
            corrected_path, b0_path, mask_path = correct_b0(
                combined_csi_path=input_path,
                output_dir=temporary_directory,
                dat_path=dat_path,
                julia_executable=julia_executable,
                julia_project=julia_project,
            )

            fid = np.load(corrected_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)

            print(
                f"[infer_combined_csi] B0 map: {b0_path}"
            )

            saved_path = _infer_and_save_combined_csi(
                input_path=input_path,
                output_path=output_path,
                fid=fid,
                mask=mask,
                model_dir=model_dir,
                fid_axis=fid_axis,
                checkpoint=checkpoint,
                batch_size=batch_size,
                device=device,
                eps=eps,
                replace_mask=True,
            )

        return saved_path

    print(
        f"[infer_combined_csi] Loading without B0 correction: {input_path}"
    )

    fid, mask = load_combined_csi(input_path)

    return _infer_and_save_combined_csi(
        input_path=input_path,
        output_path=output_path,
        fid=fid,
        mask=mask,
        model_dir=model_dir,
        fid_axis=fid_axis,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=device,
        eps=eps,
        replace_mask=False,
    )


def _infer_and_save_combined_csi(
    *,
    input_path: Path,
    output_path: Path,
    fid: np.ndarray,
    mask: np.ndarray,
    model_dir: PathLike,
    fid_axis: Union[int, str],
    checkpoint: str,
    batch_size: int,
    device: Union[str, torch.device, None],
    eps: float,
    replace_mask: bool,
) -> Path:
    """Validate prepared CombinedCSI arrays, infer, and save the MAT copy."""
    fid = np.asarray(fid)
    mask = np.asarray(mask)

    print(
        f"[infer_combined_csi] csi.Data shape: {fid.shape}"
    )
    print(
        f"[infer_combined_csi] mask shape: {mask.shape}"
    )

    if fid_axis == "auto":
        resolved_fid_axis = int(
            np.argmax(fid.shape)
        )

    elif isinstance(
        fid_axis,
        (int, np.integer),
    ):
        resolved_fid_axis = int(
            fid_axis
        )

        if resolved_fid_axis < 0:
            resolved_fid_axis += fid.ndim

        if not 0 <= resolved_fid_axis < fid.ndim:
            raise np.AxisError(
                fid_axis,
                ndim=fid.ndim,
            )

    else:
        raise TypeError(
            "fid_axis must be 'auto' or an integer."
        )

    expected_mask_shape = (
        fid.shape[:resolved_fid_axis]
        + fid.shape[resolved_fid_axis + 1 :]
    )

    if mask.shape != expected_mask_shape:
        raise ValueError(
            f"CombinedCSI mask has shape {mask.shape}, but csi.Data "
            f"with FID axis {resolved_fid_axis} requires "
            f"mask shape {expected_mask_shape}."
        )

    print(
        f"[infer_combined_csi] Running WALINET with FID axis "
        f"{resolved_fid_axis}."
    )

    cleaned_fid = infer_fid(
        fid=fid,
        model_dir=model_dir,
        output_path=None,
        fid_axis=resolved_fid_axis,
        headmask=mask,
        checkpoint=checkpoint,
        batch_size=batch_size,
        device=device,
        eps=eps,
    )

    print(
        f"[infer_combined_csi] Cleaned csi.Data shape: "
        f"{cleaned_fid.shape}"
    )

    saved_path = save_combined_csi(
        input_path=input_path,
        output_path=output_path,
        data=cleaned_fid,
        mask=mask if replace_mask else None,
    )

    print(
        f"[infer_combined_csi] Saved WALINET output: "
        f"{saved_path}"
    )

    return saved_path
