import os
import sys
import time
import ast
from pathlib import Path
import h5py

import numpy as np
import torch


def _parse_params_txt(params_path):
    params = {}

    with open(params_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()

            try:
                params[key] = ast.literal_eval(val)
            except Exception:
                params[key] = val

    return params


def _ensure_repo_src_on_path():
    candidates = [
        Path.cwd() / "src",
        Path.cwd().parent / "src",
    ]

    for p in candidates:
        if p.exists():
            sys.path.insert(0, str(p.resolve()))


def _load_model_and_params(exp, model_root="../models", architecture="auto"):
    model_dir = Path(model_root) / exp

    # New refactored model folder: params.txt + current walinet package
    params_path = model_dir / "params.txt"

    if params_path.exists():
        params = _parse_params_txt(params_path)

        _ensure_repo_src_on_path()
        from walinet.model.model import yModel, uModel

        architecture = _resolve_architecture(params, architecture)

        if architecture == "ynet":
            model_cls = yModel
        elif architecture == "unet":
            model_cls = uModel
        else:
            raise ValueError(
                f"Unknown architecture '{architecture}'. "
                "Use 'ynet', 'unet', or 'auto'."
            )

        return model_cls, params, architecture, model_dir

    # Legacy fallback: old copied src/config structure
    for sub in os.listdir(model_dir):
        if sub.startswith("src"):
            sys.path.insert(0, str(model_dir.resolve()))
            sys.path.insert(0, str((model_dir / sub).resolve()))

            from config import params
            from src.model import yModel

            try:
                from src.model import uModel
            except Exception:
                uModel = None

            if architecture == "auto":
                architecture = params.get("architecture", "ynet")

            architecture = architecture.lower()

            if architecture == "ynet":
                model_cls = yModel
            elif architecture == "unet":
                if uModel is None:
                    raise ImportError("Legacy source does not contain uModel.")
                model_cls = uModel
            else:
                raise ValueError(
                    f"Unknown architecture '{architecture}'. "
                    "Use 'ynet', 'unet', or 'auto'."
                )

            return model_cls, params, architecture, model_dir

    raise FileNotFoundError(
        f"Could not find params.txt or legacy src snapshot in {model_dir}"
    )


def runNNLipRemoval2(
    device,
    exp,
    spectra,
    LipidProj_Operator_ff=None,
    headmask=None,
    batch_size=200,
    normalization="auto",
    architecture="auto",
    model_root="../models",
    checkpoint="model_best.pt",
    eps=1e-8,
):
    """
    Run trained WALINET/Y-Net or U-Net model for nuisance removal.

    Backward compatible:
        - Y-Net uses spectra + lipid projection input.
        - U-Net uses spectra only.

    Important:
        - ynet always needs LipidProj_Operator_ff.
        - unet + projection_energy still needs LipidProj_Operator_ff for normalization.
        - unet + max_abs does not need LipidProj_Operator_ff.
    """

    t0 = time.time()

    X, Y, Z, T = spectra.shape
    Nvox = X * Y * Z

    model_cls, params, architecture, model_dir = _load_model_and_params(
        exp=exp,
        model_root=model_root,
        architecture=architecture,
    )

    normalization = _resolve_normalization(params, normalization)

    print(f"[runNNLipRemoval2] Experiment: {exp}")
    print(f"[runNNLipRemoval2] Architecture: {architecture}")
    print(f"[runNNLipRemoval2] Normalization: {normalization}")

    needs_lipid_projection = (
        architecture == "ynet"
        or normalization == "projection_energy"
    )

    if needs_lipid_projection and LipidProj_Operator_ff is None:
        raise ValueError(
            "LipidProj_Operator_ff is required for either ynet inference "
            "or projection_energy normalization.\n"
            "For fully operator-free inference use: architecture='unet' "
            "and normalization='max_abs'."
        )

    # Flatten spectra
    S_flat = spectra.reshape(Nvox, T)

    if headmask is None:
        mask_flat = np.ones(Nvox, dtype=bool)
    else:
        mask_flat = headmask.flatten() > 0

    selected = np.where(mask_flat)[0]
    lip_arr = S_flat[selected, :]

    if LipidProj_Operator_ff is not None:
        lipProj_arr = lip_arr.dot(LipidProj_Operator_ff)
    else:
        lipProj_arr = None

    # Convert to torch
    lip_t = torch.tensor(lip_arr, dtype=torch.cfloat, device=device)

    if lipProj_arr is not None:
        lipProj_t = torch.tensor(lipProj_arr, dtype=torch.cfloat, device=device)
    else:
        lipProj_t = None

    # Filter NaNs
    valid_mask = ~torch.isnan(lip_t).any(dim=1)

    if lipProj_t is not None:
        valid_mask = valid_mask & ~torch.isnan(lipProj_t).any(dim=1)

    lip_t = lip_t[valid_mask]

    if lipProj_t is not None:
        lipProj_t = lipProj_t[valid_mask]

    valid_idx = np.where(valid_mask.cpu().numpy())[0]

    # Instantiate model
    model = model_cls(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    )

    ckpt_path = model_dir / checkpoint

    if not ckpt_path.exists():
        fallback = model_dir / "model_last.pt"
        print(
            f"[runNNLipRemoval2] Checkpoint {ckpt_path.name} not found. "
            f"Falling back to {fallback.name}."
        )
        ckpt_path = fallback

    print(f"[runNNLipRemoval2] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # Normalization
    if normalization == "projection_energy":
        norm = torch.sqrt(
            torch.sum((lip_t - lipProj_t).abs() ** 2, dim=1) + eps
        )[:, None]

    elif normalization == "max_abs":
        norm = torch.amax(torch.abs(lip_t), dim=1, keepdim=True)

    else:
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            "Use 'projection_energy', 'max_abs', or 'auto'."
        )

    norm = torch.clamp(norm, min=eps)

    # Pack real/imag input
    lip_norm = lip_t / norm
    lip_in = torch.stack((lip_norm.real, lip_norm.imag), dim=1)

    if architecture == "ynet":
        lipProj_norm = lipProj_t / norm
        lipProj_in = torch.stack(
            (lipProj_norm.real, lipProj_norm.imag),
            dim=1,
        )
    else:
        lipProj_in = None

    # Inference
    Nsel = lip_t.shape[0]
    preds = torch.zeros((Nsel, 2, T), dtype=torch.float32, device="cpu")

    with torch.no_grad():
        for i in range(0, Nsel, batch_size):
            lb = lip_in[i:i + batch_size].to(device)

            if architecture == "ynet":
                lp = lipProj_in[i:i + batch_size].to(device)
                out = model(lb, lp)[:, :2, :]
            elif architecture == "unet":
                out = model(lb)[:, :2, :]
            else:
                raise ValueError(f"Unknown architecture '{architecture}'.")

            preds[i:i + batch_size] = out.detach().cpu()

    # Reconstruct complex nuisance prediction and undo normalization
    pred_c = preds[:, 0, :] + 1j * preds[:, 1, :]
    pred_c = pred_c * norm.cpu()

    # Subtract predicted nuisance from original spectrum
    removed = lip_t.cpu() - pred_c

    # Scatter back
    full = np.zeros((Nvox, T), dtype=np.complex128)
    full_indices = selected[valid_idx]
    full[full_indices, :] = removed.numpy()

    out_vol = full.reshape(X, Y, Z, T)

    print(f"[runNNLipRemoval2] done in {time.time() - t0:.1f}s")

    return out_vol

def runNNOnTrainDataH5(
    device,
    exp,
    h5_path,
    batch_size=8192,
    normalization="auto",
    architecture="auto",
    model_root="../models",
    checkpoint="model_best.pt",
    eps=1e-8,
    return_all=False,
):
    """
    Run trained WALINET/Y-Net or U-Net on simulated TrainData_*.h5.

    Expected HDF5 keys:
        spectra      (N, T), complex
        lipid_proj   (N, T), complex, required for ynet or projection_energy
        lipid        (N, T), complex, optional for return_all
        water        (N, T), complex, optional for return_all

    Returns:
        pred_nuisance:
            Complex model output in original amplitude scale,
            shape (N, T).

    If return_all=True:
        returns dict with spectra, pred_nuisance, target_nuisance,
        clean_pred, clean_gt, etc.
    """

    t0 = time.time()
    h5_path = Path(h5_path)

    model_cls, params, architecture, model_dir = _load_model_and_params(
        exp=exp,
        model_root=model_root,
        architecture=architecture,
    )

    normalization = _resolve_normalization(params, normalization)

    print(f"[runNNOnTrainDataH5] Experiment: {exp}")
    print(f"[runNNOnTrainDataH5] Architecture: {architecture}")
    print(f"[runNNOnTrainDataH5] Normalization: {normalization}")
    print(f"[runNNOnTrainDataH5] H5: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        spectra = f["spectra"][:]

        if "lipid_proj" in f:
            lipid_proj = f["lipid_proj"][:]
        else:
            lipid_proj = None

        if return_all:
            lipid = f["lipid"][:] if "lipid" in f else None
            water = f["water"][:] if "water" in f else None
        else:
            lipid = None
            water = None

    if spectra.ndim != 2:
        raise ValueError(
            f"spectra must have shape (N, T), got {spectra.shape}"
        )

    N, T = spectra.shape

    needs_lipid_projection = (
        architecture == "ynet"
        or normalization == "projection_energy"
    )

    if needs_lipid_projection and lipid_proj is None:
        raise ValueError(
            "lipid_proj is required for ynet inference or "
            "projection_energy normalization."
        )

    if lipid_proj is not None and lipid_proj.shape != spectra.shape:
        raise ValueError(
            f"lipid_proj must have shape {spectra.shape}, "
            f"got {lipid_proj.shape}"
        )

    # Instantiate model
    model = model_cls(
        nLayers=int(params["nLayers"]),
        nFilters=int(params["nFilters"]),
        dropout=float(params.get("dropout", 0.0)),
        in_channels=int(params["in_channels"]),
        out_channels=int(params["out_channels"]),
    )

    ckpt_path = model_dir / checkpoint

    if not ckpt_path.exists():
        fallback = model_dir / "model_last.pt"
        print(
            f"[runNNOnTrainDataH5] Checkpoint {ckpt_path.name} not found. "
            f"Falling back to {fallback.name}."
        )
        ckpt_path = fallback

    print(f"[runNNOnTrainDataH5] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    model.load_state_dict(ckpt)
    model.to(device).eval()

    # Valid rows
    valid_mask = np.isfinite(spectra).all(axis=1)

    if lipid_proj is not None:
        valid_mask &= np.isfinite(lipid_proj).all(axis=1)

    valid_indices = np.where(valid_mask)[0]

    pred_nuisance = np.full(
        spectra.shape,
        np.nan + 1j * np.nan,
        dtype=np.complex64,
    )

    with torch.no_grad():
        for start in range(0, len(valid_indices), batch_size):
            batch_idx = valid_indices[start:start + batch_size]

            s_np = spectra[batch_idx]

            s = torch.as_tensor(
                s_np,
                dtype=torch.cfloat,
                device=device,
            )

            if lipid_proj is not None:
                lp_np = lipid_proj[batch_idx]
                lp = torch.as_tensor(
                    lp_np,
                    dtype=torch.cfloat,
                    device=device,
                )
            else:
                lp = None

            # Normalization
            if normalization == "projection_energy":
                norm = torch.sqrt(
                    torch.sum(torch.abs(s - lp) ** 2, dim=1, keepdim=True)
                    + eps
                )

            elif normalization == "max_abs":
                norm = torch.amax(
                    torch.abs(s),
                    dim=1,
                    keepdim=True,
                )

            else:
                raise ValueError(
                    f"Unknown normalization '{normalization}'. "
                    "Use 'projection_energy', 'max_abs', or 'auto'."
                )

            norm = torch.clamp(norm, min=eps)

            # Pack real/imag input
            s_norm = s / norm
            s_in = torch.stack(
                (s_norm.real, s_norm.imag),
                dim=1,
            )

            if architecture == "ynet":
                lp_norm = lp / norm
                lp_in = torch.stack(
                    (lp_norm.real, lp_norm.imag),
                    dim=1,
                )

                out = model(s_in, lp_in)[:, :2, :]

            elif architecture == "unet":
                out = model(s_in)[:, :2, :]

            else:
                raise ValueError(f"Unknown architecture '{architecture}'.")

            # Complex model output and undo normalization
            pred = torch.complex(out[:, 0, :], out[:, 1, :])
            pred = pred * norm

            pred_nuisance[batch_idx, :] = (
                pred.detach().cpu().numpy().astype(np.complex64)
            )

    print(f"[runNNOnTrainDataH5] done in {time.time() - t0:.1f}s")

    if not return_all:
        return pred_nuisance

    if lipid is None:
        target_nuisance = None
        clean_gt = None
    else:
        if water is None:
            target_nuisance = lipid
        else:
            target_nuisance = lipid + water

        clean_gt = spectra - target_nuisance

    clean_pred = spectra - pred_nuisance

    return {
        "spectra": spectra,
        "lipid_proj": lipid_proj,
        "lipid": lipid,
        "water": water,
        "target_nuisance": target_nuisance,
        "pred_nuisance": pred_nuisance,
        "clean_pred": clean_pred,
        "clean_gt": clean_gt,
        "architecture": architecture,
        "normalization": normalization,
        "exp": exp,
        "h5_path": str(h5_path),
    }

def _resolve_architecture(params, architecture="auto"):
    """
    Resolve architecture.

    New models:
        read from params["architecture"]

    Legacy models / missing key:
        default to "ynet"
    """
    if architecture is None or architecture == "auto":
        architecture = params.get("architecture", "ynet")

    architecture = str(architecture).lower()

    if architecture not in ("ynet", "unet"):
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Use 'ynet', 'unet', or 'auto'."
        )

    return architecture


def _resolve_normalization(params, normalization="auto"):
    """
    Resolve normalization.

    New models:
        read from params["normalization"]

    Legacy models / missing key:
        default to "projection_energy"
    """
    if normalization is None or normalization == "auto":
        normalization = params.get("normalization", "projection_energy")

    normalization = str(normalization).lower()

    if normalization not in ("projection_energy", "max_abs"):
        raise ValueError(
            f"Unknown normalization '{normalization}'. "
            "Use 'projection_energy', 'max_abs', or 'auto'."
        )

    return normalization