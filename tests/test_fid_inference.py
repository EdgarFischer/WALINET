import numpy as np
import torch

from walinet.inference import fid_inference
from walinet.inference.fid_inference import (
    AcquisitionInfo,
    _fid_to_spectrum,
    _prepare_fid_length,
    _spectrum_to_fid,
)


def test_variable_length_training_keeps_length_inside_range():
    acquisition = AcquisitionInfo(840, 420, 840, False)
    fid = np.ones((2, 558, 3), dtype=np.complex64)

    prepared = _prepare_fid_length(np.moveaxis(fid, 1, -1), acquisition)

    assert prepared.shape == (2, 3, 558)


def test_missing_length_metadata_keeps_input_unchanged():
    fid = np.ones((2, 558), dtype=np.complex64)

    prepared = _prepare_fid_length(fid, None)

    assert prepared is fid


def test_variable_length_training_pads_to_minimum():
    acquisition = AcquisitionInfo(840, 420, 840, False)
    fid = np.ones((2, 300), dtype=np.complex64)

    prepared = _prepare_fid_length(fid, acquisition)

    assert prepared.shape == (2, 420)
    np.testing.assert_array_equal(prepared[:, :300], fid)
    np.testing.assert_array_equal(prepared[:, 300:], 0)


def test_variable_length_training_crops_to_maximum():
    acquisition = AcquisitionInfo(840, 420, 840, False)
    fid = np.arange(900, dtype=np.complex64)[None, :]

    prepared = _prepare_fid_length(fid, acquisition)

    np.testing.assert_array_equal(prepared, fid[:, :840])


def test_zero_filled_training_always_uses_fixed_length():
    acquisition = AcquisitionInfo(840, 420, 840, True)

    assert _prepare_fid_length(np.ones((2, 558)), acquisition).shape == (2, 840)
    assert _prepare_fid_length(np.ones((2, 900)), acquisition).shape == (2, 840)


def test_fft_roundtrip_preserves_complex_fid():
    generator = np.random.default_rng(42)
    fid = generator.normal(size=(2, 3, 558)) + 1j * generator.normal(
        size=(2, 3, 558)
    )

    reconstructed = _spectrum_to_fid(_fid_to_spectrum(fid))

    np.testing.assert_allclose(reconstructed, fid, rtol=1e-12, atol=1e-12)


def test_moved_fid_axis_can_be_restored_to_original_position():
    fid = np.ones((2, 558, 3), dtype=np.complex64)

    internal = np.moveaxis(fid, 1, -1)
    restored = np.moveaxis(internal, -1, 1)

    assert restored.shape == fid.shape
    np.testing.assert_array_equal(restored, fid)


def test_infer_fid_accepts_numpy_paths_and_saves_output(tmp_path, monkeypatch):
    class ZeroNuisanceModel(torch.nn.Module):
        def __init__(self, **kwargs):
            super().__init__()

        def forward(self, spectra):
            return torch.zeros_like(spectra)

    fid = np.ones((2, 3, 8), dtype=np.complex64)
    mask = np.ones((2, 3), dtype=bool)
    fid_path = tmp_path / "data.npy"
    mask_path = tmp_path / "mask.npy"
    output_path = tmp_path / "clean.npy"
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model_best.pt").touch()
    np.save(fid_path, fid)
    np.save(mask_path, mask)

    monkeypatch.setattr(
        fid_inference,
        "_load_model_and_params",
        lambda **kwargs: (
            ZeroNuisanceModel,
            {
                "nLayers": 1,
                "nFilters": 1,
                "in_channels": 2,
                "out_channels": 2,
                "normalization": "max_abs",
            },
            "unet",
            model_dir,
        ),
    )
    monkeypatch.setattr(
        fid_inference, "_load_checkpoint_state_dict", lambda *args, **kwargs: {}
    )

    result = fid_inference.infer_fid(
        fid=fid_path,
        headmask=mask_path,
        model_dir=model_dir,
        output_path=output_path,
        fid_axis=-1,
        device="cpu",
    )

    np.testing.assert_allclose(result, fid, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(np.load(output_path), fid, rtol=1e-6, atol=1e-6)
