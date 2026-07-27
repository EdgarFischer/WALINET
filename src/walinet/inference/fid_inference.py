from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import yaml

from walinet.inference.inference import (
    _load_checkpoint_state_dict,
    _load_model_and_params,
    _resolve_normalization,
)


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


def _load_acquisition_info(model_dir: Path) -> AcquisitionInfo | None:
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
        paths = "\n".join(f"  {path}" for path, _ in candidates)
        raise ValueError(
            "Multiple saved simulation configs contain acquisition lengths:\n"
            f"{paths}"
        )

    _, acquisition = candidates[0]
    return AcquisitionInfo(
        n_timepoints=int(acquisition["n_timepoints"]),
        min_n_timepoints=int(acquisition["min_acquired_n_timepoints"]),
        max_n_timepoints=int(acquisition["max_acquired_n_timepoints"]),
        # This is also the default used by the simulation config loader.
        zero_filling=bool(acquisition.get("zero_filling", True)),
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
            max(input_length, acquisition.min_n_timepoints),
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
        padding[-1] = (0, target_length - input_length)
        return np.pad(fid, padding, mode="constant")

    return fid


def _fid_to_spectrum(fid: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft(fid, axis=-1), axes=-1)


def _spectrum_to_fid(spectrum: np.ndarray) -> np.ndarray:
    return np.fft.ifft(
        np.fft.ifftshift(spectrum, axes=-1),
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
    """Run max-abs-normalized U-Net inference on fft-shifted spectra."""
    spatial_shape = spectra.shape[:-1]
    n_timepoints = spectra.shape[-1]
    flat = spectra.reshape(-1, n_timepoints)

    if headmask is None:
        selected = np.arange(flat.shape[0])
    else:
        headmask = np.asarray(headmask)
        if headmask.shape != spatial_shape:
            raise ValueError(
                f"headmask has shape {headmask.shape}, expected {spatial_shape}."
            )
        selected = np.flatnonzero(headmask.reshape(-1) > 0)

    selected_spectra = flat[selected]
    valid = np.isfinite(selected_spectra).all(axis=1)
    valid_indices = selected[valid]
    selected_spectra = selected_spectra[valid]

    clean = np.zeros(flat.shape, dtype=np.complex64)
    model.to(device).eval()

    with torch.no_grad():
        for start in range(0, len(selected_spectra), batch_size):
            batch_np = selected_spectra[start : start + batch_size]
            batch = torch.as_tensor(batch_np, dtype=torch.cfloat, device=device)
            norm = torch.amax(torch.abs(batch), dim=1, keepdim=True)
            norm = torch.clamp(norm, min=eps)
            normalized = batch / norm
            network_input = torch.stack(
                (normalized.real, normalized.imag),
                dim=1,
            )
            output = model(network_input)[:, :2, :]
            nuisance = torch.complex(output[:, 0, :], output[:, 1, :]) * norm
            clean_batch = batch - nuisance
            clean[valid_indices[start : start + batch_size]] = (
                clean_batch.cpu().numpy().astype(np.complex64)
            )

    return clean.reshape(*spatial_shape, n_timepoints)


def infer_fid(
    fid: np.ndarray,
    model_dir: Union[str, Path],
    *,
    output_path: Union[str, Path, None] = None,
    fid_axis: Union[int, str] = "auto",
    headmask: np.ndarray | None = None,
    checkpoint: str = "model_best.pt",
    batch_size: int = 200,
    device: Union[str, torch.device, None] = None,
    eps: float = 1e-8,
) -> np.ndarray:
    """Remove nuisance signals from complex FIDs with a trained U-Net."""
    if not isinstance(fid, np.ndarray):
        raise TypeError("fid must be a numpy.ndarray.")
    if fid.ndim == 0:
        raise ValueError("fid must have at least one dimension.")
    if not np.issubdtype(fid.dtype, np.number):
        raise TypeError("fid must contain numeric values.")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    if fid_axis == "auto":
        original_axis = int(np.argmax(fid.shape))
    elif isinstance(fid_axis, (int, np.integer)):
        original_axis = int(fid_axis)
        if original_axis < 0:
            original_axis += fid.ndim
        if not 0 <= original_axis < fid.ndim:
            raise np.AxisError(fid_axis, ndim=fid.ndim)
    else:
        raise TypeError("fid_axis must be 'auto' or an integer.")

    model_dir = Path(model_dir).expanduser().resolve()
    model_cls, params, architecture, loaded_dir = _load_model_and_params(
        exp=model_dir.name,
        model_root=model_dir.parent,
        architecture="auto",
    )
    normalization = _resolve_normalization(params, "auto")

    if architecture != "unet" or normalization != "max_abs":
        raise ValueError(
            "infer_fid currently supports operator-free U-Nets with "
            "max_abs normalization only; found "
            f"architecture={architecture!r}, normalization={normalization!r}."
        )

    acquisition = _load_acquisition_info(loaded_dir)
    prepared = np.moveaxis(np.asarray(fid, dtype=np.complex64), original_axis, -1)
    prepared = _prepare_fid_length(prepared, acquisition)
    spectra = _fid_to_spectrum(prepared)

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = model_cls(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    )
    checkpoint_path = loaded_dir / checkpoint
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    model.load_state_dict(_load_checkpoint_state_dict(checkpoint_path, device))

    clean_spectra = _infer_spectra(
        spectra,
        model=model,
        device=device,
        batch_size=batch_size,
        headmask=headmask,
        eps=eps,
    )
    clean_fid = _spectrum_to_fid(clean_spectra)
    clean_fid = np.moveaxis(clean_fid, -1, original_axis)

    if output_path is not None:
        output_path = Path(output_path)
        if output_path.suffix.lower() != ".npy":
            raise ValueError("Only .npy output is currently supported.")
        np.save(output_path, clean_fid)

    return clean_fid
