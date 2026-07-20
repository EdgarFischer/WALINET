from collections.abc import Sequence
from pathlib import Path

import nibabel as nib
import numpy as np


def load_subject_maps(
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    extension: str = ".nii.gz",
    *,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Load the same image map for multiple subjects and stack subjects
    along the last array dimension.

    Parameters
    ----------
    base_paths:
        Subject-specific base directories. Their order determines the
        order along the final subject dimension.

    relative_path:
        Path to the map relative to each base directory, without the
        file extension.

        Example:
            "Extra/FWHM_map"

    extension:
        File extension, for example ".nii.gz" or ".nii".

    dtype:
        NumPy dtype of the returned array.

    Returns
    -------
    np.ndarray
        Array with shape:

            (*spatial_shape, n_subjects)

        For example:

            (64, 64, 35, 9)

    Raises
    ------
    FileNotFoundError:
        If one of the requested files does not exist.

    ValueError:
        If no base paths were supplied or the image shapes differ.
    """
    base_paths = [
        Path(base_path)
        for base_path in base_paths
    ]

    if not base_paths:
        raise ValueError(
            "base_paths must contain at least one subject directory."
        )

    if not extension.startswith("."):
        extension = f".{extension}"

    relative_file = Path(
        f"{relative_path}{extension}"
    )

    subject_arrays: list[np.ndarray] = []
    expected_shape: tuple[int, ...] | None = None

    for base_path in base_paths:
        file_path = base_path / relative_file

        if not file_path.is_file():
            raise FileNotFoundError(
                "Subject map not found:\n"
                f"  {file_path}"
            )

        image = nib.load(
            str(file_path)
        )

        array = np.asarray(
            image.get_fdata(),
            dtype=dtype,
        )

        if expected_shape is None:
            expected_shape = array.shape

        elif array.shape != expected_shape:
            raise ValueError(
                "All subject maps must have the same shape:\n"
                f"  expected: {expected_shape}\n"
                f"  found:    {array.shape}\n"
                f"  file:     {file_path}"
            )

        subject_arrays.append(
            array
        )

    return np.stack(
        subject_arrays,
        axis=-1,
    )