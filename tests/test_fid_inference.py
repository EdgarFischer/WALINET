import numpy as np

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
