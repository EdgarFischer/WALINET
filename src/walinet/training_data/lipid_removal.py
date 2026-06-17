import logging
import numpy as np


def diag_mean(beta: float, M: np.ndarray) -> float:
    """Mean absolute diagonal of (I + beta * M)^-1."""
    identity = np.eye(M.shape[0], dtype=M.dtype)
    rem_op = np.linalg.inv(identity + beta * M)
    return float(np.mean(np.abs(np.diag(rem_op))))


def find_beta_bisect(
    lipid_rf: np.ndarray,
    target: float = 0.938,
    tol: float = 5e-3,
    max_iter: int = 60,
):
    """
    Bracketing/bisection for beta such that
    abs(diag_mean(beta) - target) <= tol.

    CAUTION: Do not change target=0.938 unless you know what you are doing,
    because WALINET was trained with this value.
    """
    M = lipid_rf.conj().T @ lipid_rf

    beta_low = 0.0
    beta_high = 1e-10

    g_high = diag_mean(beta_high, M)

    while g_high > target:
        beta_high *= 2.0
        g_high = diag_mean(beta_high, M)

        if beta_high > 1e12:
            raise RuntimeError(
                "Bracket search failed. Target may be larger than diag_mean(0)."
            )

    for _ in range(max_iter):
        beta_mid = 0.5 * (beta_low + beta_high)
        g_mid = diag_mean(beta_mid, M)

        if abs(g_mid - target) <= tol:
            return beta_mid, M

        if g_mid > target:
            beta_low = beta_mid
        else:
            beta_high = beta_mid

    raise RuntimeError("Desired tolerance was not reached.")


def compute_lipid_projection_operator(
    spectra: np.ndarray,
    lipid_mask: np.ndarray,
    max_n_iter: int = 60,
    target: float = 0.938,
    tol: float = 5e-3,
):
    """
    Args:
        spectra: (X, Y, Z, T) complex spectra, already FFT-transformed.
        lipid_mask: (X, Y, Z) boolean or 0/1 lipid mask.

    Returns:
        LipidProj_Operator_ff: (T, T)
    """
    T = spectra.shape[-1]

    data_rf = spectra.reshape(-1, T)
    lipid_rf = data_rf[lipid_mask.flatten() > 0, :]

    beta, M = find_beta_bisect(
        lipid_rf,
        target=target,
        tol=tol,
        max_iter=max_n_iter,
    )

    identity = np.eye(T, dtype=M.dtype)
    rem_op = np.linalg.inv(identity + beta * M)
    lipid_proj_operator_ff = identity - rem_op

    fac = diag_mean(beta, M)
    msg = f"Achieved diag-mean = {fac:.4f} at beta = {beta:.2e}"
    logging.info(msg)
    print(msg)

    return lipid_proj_operator_ff