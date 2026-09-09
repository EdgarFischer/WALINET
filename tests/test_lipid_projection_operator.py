import numpy as np
import pytest
import torch

from walinet.training_data.lipid_removal import (
    compute_lipid_projection_operator,
    find_beta_bisect,
)


def test_eigenvalue_beta_search_matches_legacy() -> None:
    rng = np.random.default_rng(123)
    lipid_rf = (
        rng.normal(size=(20, 12))
        + 1j * rng.normal(size=(20, 12))
    ).astype(np.complex64)

    beta_legacy, matrix_legacy = find_beta_bisect(
        lipid_rf,
        method="legacy",
    )
    beta_fast, matrix_fast = find_beta_bisect(
        lipid_rf,
        method="eigenvalue",
    )

    assert beta_fast == pytest.approx(beta_legacy, rel=1e-6)
    np.testing.assert_array_equal(matrix_fast, matrix_legacy)


def test_fast_projection_operator_matches_legacy() -> None:
    rng = np.random.default_rng(456)
    spectra = (
        rng.normal(size=(4, 2, 1, 12))
        + 1j * rng.normal(size=(4, 2, 1, 12))
    ).astype(np.complex64)
    lipid_mask = np.ones(spectra.shape[:-1], dtype=bool)

    legacy = compute_lipid_projection_operator(
        spectra,
        lipid_mask,
        method="legacy",
    )
    fast = compute_lipid_projection_operator(
        spectra,
        lipid_mask,
        method="eigenvalue",
    )

    np.testing.assert_allclose(fast, legacy, rtol=1e-5, atol=1e-6)


def test_unknown_beta_search_method_is_rejected() -> None:
    with pytest.raises(ValueError, match="method"):
        find_beta_bisect(
            np.eye(4, dtype=np.complex64),
            method="unknown",
        )


def test_default_method_falls_back_to_cpu_without_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    lipid_rf = np.eye(4, dtype=np.complex64)

    beta_auto, matrix_auto = find_beta_bisect(lipid_rf)
    beta_cpu, matrix_cpu = find_beta_bisect(
        lipid_rf,
        method="eigenvalue",
    )

    assert beta_auto == beta_cpu
    np.testing.assert_array_equal(matrix_auto, matrix_cpu)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is unavailable",
)
def test_gpu_projection_operator_matches_legacy() -> None:
    rng = np.random.default_rng(789)
    spectra = (
        rng.normal(size=(4, 2, 1, 12))
        + 1j * rng.normal(size=(4, 2, 1, 12))
    ).astype(np.complex64)
    lipid_mask = np.ones(spectra.shape[:-1], dtype=bool)

    legacy = compute_lipid_projection_operator(
        spectra,
        lipid_mask,
        method="legacy",
    )
    gpu = compute_lipid_projection_operator(
        spectra,
        lipid_mask,
        method="gpu",
        gpu_index=0,
    )

    np.testing.assert_allclose(gpu, legacy, rtol=1e-4, atol=1e-5)
