import h5py
import numpy as np
import torch

from walinet.data.dataloader import SpectrumDatasetLoad


def _write_train_h5(path):
    spectra = np.array(
        [
            [1 + 1j, 2 + 0j, 0 + 1j, 4 + 0j],
            [2 + 0j, 0 + 2j, 1 + 0j, 1 + 1j],
        ],
        dtype=np.complex64,
    )

    lipid = 0.1 * spectra
    water = 0.2 * spectra
    lipid_proj = 0.5 * spectra

    with h5py.File(path, "w") as f:
        f.create_dataset("spectra", data=spectra)
        f.create_dataset("lipid", data=lipid)
        f.create_dataset("water", data=water)
        f.create_dataset("lipid_proj", data=lipid_proj)

    return spectra, lipid, water, lipid_proj


def _make_dataset(tmp_path, normalization):
    subject = "sub01"
    train_dir = tmp_path / subject / "TrainData"
    train_dir.mkdir(parents=True)

    h5_path = train_dir / "TrainData_vtest.h5"
    spectra, lipid, water, lipid_proj = _write_train_h5(h5_path)

    params = {
        "path_to_data": str(tmp_path) + "/",
        "normalization": normalization,
    }

    dataset = SpectrumDatasetLoad(
        params=params,
        files=[subject],
        version="vtest",
        aug=False,
    )

    return dataset, spectra, lipid, water, lipid_proj


def test_dataloader_max_abs_normalization(tmp_path):
    dataset, spectra, lipid, water, lipid_proj = _make_dataset(
        tmp_path,
        normalization="max_abs",
    )

    norm = np.max(np.abs(spectra), axis=-1)

    expected_spectra = spectra / norm[:, None]
    expected_nuisance = (lipid + water) / norm[:, None]
    expected_lipid_proj = lipid_proj / norm[:, None]

    assert torch.allclose(
        dataset.spectra,
        torch.tensor(expected_spectra, dtype=torch.cfloat),
        atol=1e-6,
    )

    assert torch.allclose(
        dataset.nuisance,
        torch.tensor(expected_nuisance, dtype=torch.cfloat),
        atol=1e-6,
    )

    assert torch.allclose(
        dataset.nuisance_proj,
        torch.tensor(expected_lipid_proj, dtype=torch.cfloat),
        atol=1e-6,
    )


def test_dataloader_projection_energy_normalization(tmp_path):
    dataset, spectra, lipid, water, lipid_proj = _make_dataset(
        tmp_path,
        normalization="projection_energy",
    )

    s1 = 0
    s2 = -1

    norm = np.sqrt(
        np.sum(
            np.abs(spectra[:, s1:s2] - lipid_proj[:, s1:s2]) ** 2,
            axis=1,
        )
    )

    expected_spectra = spectra / norm[:, None]
    expected_nuisance = (lipid + water) / norm[:, None]
    expected_lipid_proj = lipid_proj / norm[:, None]

    assert torch.allclose(
        dataset.spectra,
        torch.tensor(expected_spectra, dtype=torch.cfloat),
        atol=1e-6,
    )

    assert torch.allclose(
        dataset.nuisance,
        torch.tensor(expected_nuisance, dtype=torch.cfloat),
        atol=1e-6,
    )

    assert torch.allclose(
        dataset.nuisance_proj,
        torch.tensor(expected_lipid_proj, dtype=torch.cfloat),
        atol=1e-6,
    )