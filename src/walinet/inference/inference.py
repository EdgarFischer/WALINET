import os
import sys
import time

import numpy as np
import torch

eps = 1e-8  # Small epsilon to avoid division by zero

def runNNLipRemoval2(device, exp, spectra, LipidProj_Operator_ff, headmask, batch_size=200):
    """
    Run your trained yModel to remove lipids from `spectra`.

    Args:
        device: torch.device
        exp:    str, name of your experiment folder under ../models/
        spectra: np.ndarray, shape (X,Y,Z,T), complex after fft+fftshift
        LipidProj_Operator_ff: np.ndarray, shape (T,T), lipid‐projection operator
        headmask: np.ndarray, shape (X,Y,Z), boolean or 0/1 mask for voxels
        batch_size: int, inference batch size
    Returns:
        out_vol: np.ndarray, shape (X,Y,Z,T), lipid‐removed spectra
    """
    t0 = time.time()
    X, Y, Z, T = spectra.shape
    Nvox = X * Y * Z

    # — Flatten and select mask voxels
    S_flat    = spectra.reshape(Nvox, T)                  # (Nvox, T)
    mask_flat = (headmask.flatten() > 0)                  # (Nvox,)
    selected  = np.where(mask_flat)[0]                    # indices of voxels to process

    lip_arr      = S_flat[selected, :]                    # (Nsel, T)
    lipProj_arr  = lip_arr.dot(LipidProj_Operator_ff)     # (Nsel, T)

    # — Convert to torch complex tensors on device
    lip_t     = torch.tensor(lip_arr,     dtype=torch.cfloat, device=device)
    lipProj_t = torch.tensor(lipProj_arr, dtype=torch.cfloat, device=device)

    # — Filter out any rows with NaNs
    valid_mask = (~torch.isnan(lip_t).any(dim=1) &
                  ~torch.isnan(lipProj_t).any(dim=1))
    lip_t     = lip_t[valid_mask]
    lipProj_t = lipProj_t[valid_mask]
    valid_idx = np.where(valid_mask.cpu().numpy())[0]    # indices into `selected`

    # — Dynamically import your model code, just like in your original
    for sub in os.listdir(f'../models/{exp}'):
        if sub.startswith("src"):
            sys.path.insert(0, os.path.abspath(f'../models/{exp}'))
            sys.path.insert(0, os.path.abspath(f'../models/{exp}/{sub}'))
            from config    import params
            from src.model import yModel
            break

    # — Instantiate and load your trained yModel
    model = yModel(nLayers    = params["nLayers"],
                   nFilters   = params["nFilters"],
                   dropout    = 0,
                   in_channels  = params["in_channels"],
                   out_channels = params["out_channels"])
    params["path_to_model"] = f"../models/{exp}/"
    ckpt = torch.load(params["path_to_model"] + 'model_last.pt',
                      map_location=device)
    model.load_state_dict(ckpt)
    model.to(device).eval()

    # — Compute per-voxel energy normalization factor
    energy = torch.sqrt(torch.sum((lip_t - lipProj_t).abs()**2, dim=1) + eps)[:, None]
    energy = torch.clamp(energy, min=1e-3)

    # — Normalize and pack real/imag into channels (Nsel, 2, T)
    lip_norm     = lip_t     / energy
    lipProj_norm = lipProj_t / energy
    lip_in     = torch.stack((lip_norm.real,     lip_norm.imag),     dim=1)
    lipProj_in = torch.stack((lipProj_norm.real, lipProj_norm.imag), dim=1)

    # — Inference in batches
    Nsel = lip_t.shape[0]
    preds = torch.zeros((Nsel, 2, T), dtype=torch.float64, device='cpu')
    with torch.no_grad():
        for i in range(0, Nsel, batch_size):
            lb = lip_in[i:i+batch_size].to(device)
            lp = lipProj_in[i:i+batch_size].to(device)
            out = model(lb, lp)[:, :2, :].cpu()
            preds[i:i+batch_size] = out

    # — Reconstruct complex prediction & scale back
    pred_c = preds[:,0,:] + 1j * preds[:,1,:]   # (Nsel, T)
    pred_c = pred_c * energy.cpu()              # broadcast (Nsel,1) × (Nsel,T)

    # — Subtract predicted lipid from original
    removed = lip_t.cpu() - pred_c              # (Nsel, T)

    # — Scatter back into full volume
    full = np.zeros((Nvox, T), dtype=np.complex128)
    full_indices = selected[valid_idx]           # true flat indices in [0..Nvox)
    full[full_indices, :] = removed.numpy()
    out_vol = full.reshape(X, Y, Z, T)

    print(f"[runNNLipRemoval2] done in {time.time()-t0:.1f}s")
    return out_vol