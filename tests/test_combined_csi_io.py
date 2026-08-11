import h5py
import numpy as np
from scipy.io import loadmat, savemat

from walinet.data.combined_csi_io import load_combined_csi, save_combined_csi


def test_save_hdf5_combined_csi_can_replace_data_and_mask(tmp_path):
    input_path = tmp_path / "input.mat"
    output_path = tmp_path / "output.mat"

    with h5py.File(input_path, "w") as file:
        csi = file.create_group("csi")
        # Raw MATLAB-v7.3/HDF5 dimensions are reversed relative to the logical
        # MATLAB/NumPy arrays accepted and returned by the public functions.
        csi.create_dataset("Data", data=np.zeros((4, 3, 2), np.complex64))
        file.create_dataset("mask", data=np.zeros((3, 2), np.uint8))
        file.create_dataset("untouched", data=np.array([42], np.int32))

    data = np.ones((2, 3, 4), np.complex64) * (2 + 3j)
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    save_combined_csi(input_path, output_path, data, mask=mask)

    with h5py.File(output_path, "r") as file:
        np.testing.assert_array_equal(
            file["csi"]["Data"][:],
            np.transpose(data, (2, 1, 0)),
        )
        np.testing.assert_array_equal(
            file["mask"][:],
            mask.astype(np.uint8).T,
        )
        np.testing.assert_array_equal(file["untouched"][:], [42])

    loaded_data, loaded_mask = load_combined_csi(output_path)
    np.testing.assert_array_equal(loaded_data, data)
    np.testing.assert_array_equal(loaded_mask, mask.astype(np.uint8))


def test_save_classic_combined_csi_can_replace_data_and_mask(tmp_path):
    input_path = tmp_path / "input.mat"
    output_path = tmp_path / "output.mat"
    savemat(
        input_path,
        {
            "csi": {"Data": np.zeros((2, 3, 4), np.complex64)},
            "mask": np.zeros((2, 3), np.uint8),
            "untouched": np.array([[42]], np.int32),
        },
    )

    data = np.ones((2, 3, 4), np.complex64) * (2 + 3j)
    mask = np.array([[1, 0, 1], [0, 1, 0]], dtype=bool)
    save_combined_csi(input_path, output_path, data, mask=mask)

    result = loadmat(output_path, squeeze_me=True, struct_as_record=False)
    np.testing.assert_array_equal(result["csi"].Data, data)
    np.testing.assert_array_equal(result["mask"], mask)
    assert result["untouched"] == 42
