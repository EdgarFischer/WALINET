"""In-memory WALINET-to-forD classical fitting pipeline."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Union

import numpy as np
import torch

from walinet.inference.fid_inference import infer_fid


PathLike = Union[str, Path]


def run_walinet_ford_pipeline(
    data_path: PathLike,
    mask_path: PathLike,
    walinet_model_dir: PathLike,
    ford_config_template: PathLike,
    output_path: PathLike,
    gpu_number: int,
    *,
    fid_axis: Union[int, str] = "auto",
    walinet_checkpoint: str = "model_best.pt",
    walinet_batch_size: int = 200,
) -> Path:
    """Run WALINET inference and classical forD fitting without intermediate I/O.

    Only forD's final artifacts and a copy of the effective input config are
    written to ``output_path``. The WALINET-cleaned FID remains in memory.
    """
    data_path = _existing_npy(data_path, "data_path")
    mask_path = _existing_npy(mask_path, "mask_path")
    model_dir = _existing_directory(walinet_model_dir, "walinet_model_dir")
    config_path = _existing_file(ford_config_template, "ford_config_template")
    output_dir = Path(output_path).expanduser().resolve()

    if not isinstance(gpu_number, int) or isinstance(gpu_number, bool):
        raise TypeError("gpu_number must be an integer.")
    if gpu_number < 0:
        raise ValueError("gpu_number must be >= 0.")
    device = torch.device(f"cuda:{gpu_number}")

    print(f"[pipeline] Loading data: {data_path}", flush=True)
    print(f"[pipeline] Loading mask: {mask_path}", flush=True)
    data = np.load(data_path, allow_pickle=False)
    mask = np.asarray(np.load(mask_path, allow_pickle=False), dtype=bool)
    resolved_fid_axis = _resolve_fid_axis(fid_axis, data.shape)
    spatial_shape = data.shape[:resolved_fid_axis] + data.shape[resolved_fid_axis + 1 :]
    if data.ndim < 2 or spatial_shape != mask.shape:
        raise ValueError(
            f"Data spatial shape {spatial_shape} does not match mask shape "
            f"{mask.shape}."
        )

    print(
        f"[pipeline] Starting WALINET inference on {device} with model "
        f"{model_dir}",
        flush=True,
    )
    cleaned_fid = infer_fid(
        data,
        model_dir,
        headmask=mask,
        fid_axis=fid_axis,
        checkpoint=walinet_checkpoint,
        batch_size=walinet_batch_size,
        device=device,
        output_path=None,
    )
    cleaned_fid = np.moveaxis(cleaned_fid, resolved_fid_axis, -1)
    cleaned_fid = np.asarray(cleaned_fid, dtype=np.complex64)
    del data
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(
        f"[pipeline] WALINET inference finished; cleaned shape: "
        f"{cleaned_fid.shape}",
        flush=True,
    )

    # Imported lazily so WALINET-only use does not require forD dependencies.
    from forD.classical_fitting.Problem_regularized_standalone import Problem
    from forD.classical_fitting.config import Configuration

    output_dir.mkdir(parents=True, exist_ok=True)
    with config_path.open("r", encoding="utf-8") as handle:
        effective_config = json.load(handle)
    basis_path = _resolve_config_file(
        effective_config["io_config"].get("basis_path"),
        config_path,
        "io_config.basis_path",
    )
    effective_config["io_config"].update(
        {
            "basis_path": str(basis_path),
            "data_path": str(data_path),
            "mask_path": str(mask_path),
            "logging_path": str(output_dir),
            "saving_path": str(output_dir),
        }
    )
    effective_config["pytorch_config"]["device"] = str(device)
    used_config_path = output_dir / "fitting_config_used.json"
    with used_config_path.open("w", encoding="utf-8") as handle:
        json.dump(effective_config, handle, indent=2)
        handle.write("\n")

    config = Configuration.from_dict(effective_config)
    torch.set_default_device(device)
    torch.set_default_dtype(config.pytorch_config.default_type or torch.float32)
    if config.pytorch_config.float32_matmul_precision is not None:
        torch.set_float32_matmul_precision(
            config.pytorch_config.float32_matmul_precision
        )
    if config.pytorch_config.num_threads is not None:
        torch.set_num_threads(config.pytorch_config.num_threads)
    torch.manual_seed(0)

    print(
        f"[pipeline] Starting classical forD fitting; output: {output_dir}",
        flush=True,
    )
    problem = Problem(
        config,
        subject_data=cleaned_fid,
        subject_mask=mask,
    )
    problem._optimize()
    print("[pipeline] WALINET + forD pipeline completed successfully.", flush=True)
    return output_dir


def _existing_npy(path: PathLike, name: str) -> Path:
    result = _existing_file(path, name)
    if result.suffix.lower() != ".npy":
        raise ValueError(f"{name} must point to a .npy file: {result}")
    return result


def _existing_file(path: PathLike, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise FileNotFoundError(f"{name} does not exist: {result}")
    return result


def _existing_directory(path: PathLike, name: str) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_dir():
        raise FileNotFoundError(f"{name} does not exist: {result}")
    return result


def _resolve_config_file(
    configured_path: str | None,
    config_path: Path,
    name: str,
) -> Path:
    if not configured_path:
        raise ValueError(f"{name} must be set in {config_path}.")
    path = Path(configured_path).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [
            Path.cwd() / path,
            config_path.parent / path,
            config_path.parent.parent / path,
        ]
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file():
            print(f"[pipeline] Resolved {name}: {resolved}", flush=True)
            return resolved
    checked = "\n  ".join(str(candidate.resolve()) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve {name}; checked:\n  {checked}")


def _resolve_fid_axis(fid_axis: Union[int, str], shape: tuple[int, ...]) -> int:
    if fid_axis == "auto":
        return int(np.argmax(shape))
    if isinstance(fid_axis, (int, np.integer)):
        axis = int(fid_axis)
        if axis < 0:
            axis += len(shape)
        if not 0 <= axis < len(shape):
            raise np.AxisError(fid_axis, ndim=len(shape))
        return axis
    raise TypeError("fid_axis must be 'auto' or an integer.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run WALINET inference followed by classical forD fitting."
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--mask", required=True, type=Path)
    parser.add_argument("--walinet-model", required=True, type=Path)
    parser.add_argument("--ford-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--fid-axis", default="auto")
    parser.add_argument("--walinet-checkpoint", default="model_best.pt")
    parser.add_argument("--walinet-batch-size", default=200, type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    fid_axis = args.fid_axis
    if fid_axis != "auto":
        try:
            fid_axis = int(fid_axis)
        except ValueError as error:
            raise ValueError("--fid-axis must be 'auto' or an integer.") from error
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
    )


if __name__ == "__main__":
    main()
