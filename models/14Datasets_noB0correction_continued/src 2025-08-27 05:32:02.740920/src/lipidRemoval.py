import os
import sys
import numpy as np
import nibabel as nib
import h5py
import time
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from tqdm import tqdm
from itertools import product
import scipy
import torch
import torch.nn as nn

#  Hilfsfunktionen
def diag_mean(beta, M):
    """⟨diag⟩ von (I + β M)⁻¹  – ohne target-Abzug."""
    RemOP = np.linalg.inv(np.eye(M.shape[0]) + beta * M)
    return np.mean(np.abs(np.diag(RemOP)))

def find_beta_bisect(lipid_rf, target=0.938, tol=5e-3, max_iter=60): #CAUTION: Dont change the target value of 0.938, because WALINET was only trained with this value
    """
    Bracketing/Bisection für β → erfüllt |diag_mean(β) - target| ≤ tol.
    Liefert β und die bereits berechnete Matrix M.
    """
    M = lipid_rf.conj().T @ lipid_rf          # (T×T), positiv definit

    # ----- Bracket-Suche:  g(β_low) > target > g(β_high)
    beta_low, beta_high = 0.0, 1e-10          # klein anfangen
    g_high = diag_mean(beta_high, M)
    while g_high > target:                    # solange noch zu groß
        beta_high *= 2.0
        g_high = diag_mean(beta_high, M)
        if beta_high > 1e12:                  # reiner Sicherheitsstopp
            raise RuntimeError("Bracket-Suche schlug fehl – target evtl. > g(0)")
    
    # ----- Bisection
    for _ in range(max_iter):
        beta_mid = 0.5 * (beta_low + beta_high)
        g_mid    = diag_mean(beta_mid, M)
        if abs(g_mid - target) <= tol:
            return beta_mid, M
        if g_mid > target:       # noch zu groß  ⇒ β rauf
            beta_low = beta_mid
        else:                    # zu klein      ⇒ β runter
            beta_high = beta_mid
    raise RuntimeError("Gewünschte Toleranz nicht erreicht")


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


def runModelOnLipData(lip, lipProj, model, device):

    spectra_energy = torch.sqrt(torch.sum(torch.abs(lip-lipProj)**2, dim=1))[:,None]
    lip /= spectra_energy
    lip = torch.stack((torch.real(lip), torch.imag(lip)), axis=1)
    lipProj /= spectra_energy
    lipProj = torch.stack((torch.real(lipProj), torch.imag(lipProj)), axis=1)

    prediction = torch.zeros(lip.shape, dtype=torch.cfloat)
    datasz = lipProj.shape[0]
    batchsz = 200
    
    loss_func = nn.MSELoss()
    model.eval()
    with torch.no_grad():
        for i in range(int(datasz/batchsz)+1):
            log = 'Percent: {:.2f}%'
            percent = (i+1)/int(datasz/batchsz)*100
            print(log.format(percent), end='\r')
            lip_batch = lip[i*batchsz:(i+1)*batchsz,:,:]
            lipProj_batch = lipProj[i*batchsz:(i+1)*batchsz,:,:]
            
            lip_batch, lipProj_batch = lip_batch.to(device), lipProj_batch.to(device)
            pred = model(lip_batch, lipProj_batch).cpu()
            prediction[i*batchsz:(i+1)*batchsz,:] = pred
    
    
    #loss = loss_func(prediction, true)
    #print('Pred loss: ', loss.item())
    #loss = loss_func(lip, true)
    #print('Inp loss: ', loss.item())
    prediction = prediction[:,0] + 1j*prediction[:,1]
    prediction = prediction*spectra_energy
    lip = lip[:,0] + 1j*lip[:,1]
    lip = lip * spectra_energy
    Data_LipidRemoved_rf = lip - prediction
    
    return Data_LipidRemoved_rf, prediction