"""Load and save CombinedCSI.mat files."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Union

import h5py
import numpy as np
from scipy.io import loadmat, savemat


PathLike = Union[str, Path]


def _to_complex_array(raw):
    """
    Convert either a structured real/imag array or an already-complex
    MATLAB array into a NumPy complex array.
    """
    raw = np.asarray(raw)

    if raw.dtype.names is not None and {"real", "imag"}.issubset(raw.dtype.names):
        return raw["real"] + 1j * raw["imag"]

    return raw


def load_combined_csi(
    mat_path: PathLike,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load csi.Data and mask from CombinedCSI.mat.

    First tries HDF5 / MATLAB v7.3 loading via h5py.
    If that fails, falls back to scipy.io.loadmat for classic MATLAB files.
    """
    mat_path = Path(mat_path).expanduser().resolve()

    try:
        with h5py.File(mat_path, "r") as file:
            raw = file["csi"]["Data"][:]
            # MATLAB v7.3 stores array dimensions in reverse HDF5 order.
            # Present the same logical axis order as MATLAB and scipy.loadmat.
            data = _reverse_axes(_to_complex_array(raw))
            mask = _reverse_axes(file["mask"][:])

        print("  Loaded CombinedCSI.mat via h5py")
        return data, mask

    except OSError:
        print("  h5py failed; loading CombinedCSI.mat via scipy.io.loadmat")

        mat = loadmat(
            mat_path,
            squeeze_me=True,
            struct_as_record=False,
        )

        csi = mat["csi"]

        if hasattr(csi, "Data"):
            raw = csi.Data
        else:
            raw = csi["Data"]

        data = _to_complex_array(raw)
        mask = mat["mask"]

        return data, mask


def save_combined_csi(
    input_path: PathLike,
    output_path: PathLike,
    data: np.ndarray,
    mask: np.ndarray | None = None,
) -> Path:
    """
    Copy a CombinedCSI.mat file and replace csi.Data and, optionally, mask.

    If ``mask`` is None, the original mask is retained. All other fields and
    variables are always retained.

    MATLAB v7.3 files are copied and modified using h5py.
    Classic MATLAB files are loaded and written again using scipy.io.
    """
    input_path = Path(input_path).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    data = np.asarray(data)
    if mask is not None:
        mask = np.asarray(mask)

    if not input_path.is_file():
        raise FileNotFoundError(
            f"CombinedCSI.mat does not exist: {input_path}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if h5py.is_hdf5(input_path):
        _save_hdf5_combined_csi(
            input_path=input_path,
            output_path=output_path,
            data=data,
            mask=mask,
        )
        print(f"  Saved CombinedCSI.mat via h5py: {output_path}")

    else:
        _save_classic_combined_csi(
            input_path=input_path,
            output_path=output_path,
            data=data,
            mask=mask,
        )
        print(
            f"  Saved CombinedCSI.mat via scipy.io.savemat: "
            f"{output_path}"
        )

    return output_path


def _reverse_axes(array: np.ndarray) -> np.ndarray:
    """Reverse all axes between MATLAB-v7.3 and raw HDF5 conventions."""
    array = np.asarray(array)
    return np.transpose(array, axes=tuple(range(array.ndim - 1, -1, -1)))


def _save_hdf5_combined_csi(
    input_path: Path,
    output_path: Path,
    data: np.ndarray,
    mask: np.ndarray | None,
) -> None:
    """
    Copy a MATLAB v7.3 file and replace /csi/Data and, optionally, /mask.
    """
    if input_path != output_path:
        shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "r+") as file:
        dataset = file["csi"]["Data"]
        stored_data = _encode_for_hdf5_dataset(
            # ``data`` uses logical MATLAB/NumPy order. Raw MATLAB-v7.3 HDF5
            # datasets store those dimensions in reverse order.
            data=_reverse_axes(data),
            dtype=dataset.dtype,
        )

        if dataset.shape == stored_data.shape:
            dataset[...] = stored_data
        else:
            # The acquisition length may have changed through cropping or
            # padding. Recreate only /csi/Data and preserve its attributes.
            _replace_hdf5_dataset(dataset, stored_data, dtype=dataset.dtype)

        if mask is not None:
            mask_dataset = file["mask"]
            stored_mask = np.asarray(
                _reverse_axes(mask),
                dtype=mask_dataset.dtype,
            )

            if mask_dataset.shape == stored_mask.shape:
                mask_dataset[...] = stored_mask
            else:
                _replace_hdf5_dataset(
                    mask_dataset,
                    stored_mask,
                    dtype=mask_dataset.dtype,
                )


def _replace_hdf5_dataset(
    dataset: h5py.Dataset,
    data: np.ndarray,
    *,
    dtype: np.dtype,
) -> None:
    """Recreate one HDF5 dataset while retaining attributes and storage."""
    parent = dataset.parent
    name = dataset.name.rsplit("/", 1)[-1]
    attributes = dict(dataset.attrs.items())
    creation_options = _get_hdf5_creation_options(
        dataset=dataset,
        new_shape=data.shape,
    )

    del parent[name]
    new_dataset = parent.create_dataset(
        name,
        data=data,
        dtype=dtype,
        **creation_options,
    )

    for key, value in attributes.items():
        new_dataset.attrs[key] = value


def _encode_for_hdf5_dataset(
    data: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    """
    Convert complex NumPy data back to the representation used by /csi/Data.
    """
    dtype = np.dtype(dtype)

    # MATLAB v7.3 compound dtype:
    # dtype.names == ("real", "imag")
    if dtype.names is not None and {"real", "imag"}.issubset(dtype.names):
        stored = np.empty(data.shape, dtype=dtype)

        real_dtype = dtype.fields["real"][0]
        imag_dtype = dtype.fields["imag"][0]

        stored["real"] = np.asarray(
            np.real(data),
            dtype=real_dtype,
        )
        stored["imag"] = np.asarray(
            np.imag(data),
            dtype=imag_dtype,
        )

        return stored

    # Already stored as a normal complex HDF5 dataset.
    if np.issubdtype(dtype, np.complexfloating):
        return np.asarray(data, dtype=dtype)

    raise TypeError(
        "/csi/Data is neither a complex dataset nor a structured "
        f"real/imag dataset. Found dtype: {dtype}"
    )


def _get_hdf5_creation_options(
    dataset: h5py.Dataset,
    new_shape: tuple[int, ...],
) -> dict:
    """
    Preserve compatible HDF5 storage settings when recreating /csi/Data.
    """
    options = {}

    if dataset.chunks is not None:
        chunks_are_compatible = (
            len(dataset.chunks) == len(new_shape)
            and all(
                chunk <= size
                for chunk, size in zip(dataset.chunks, new_shape)
            )
        )

        options["chunks"] = (
            dataset.chunks
            if chunks_are_compatible
            else True
        )

    if dataset.compression is not None:
        options["compression"] = dataset.compression
        options["compression_opts"] = dataset.compression_opts

    if dataset.shuffle:
        options["shuffle"] = True

    if dataset.fletcher32:
        options["fletcher32"] = True

    return options


def _save_classic_combined_csi(
    input_path: Path,
    output_path: Path,
    data: np.ndarray,
    mask: np.ndarray | None,
) -> None:
    """
    Replace csi.Data in a classic, non-HDF5 MATLAB file.
    """
    # Keep dimensions and MATLAB structs intact while rewriting the file.
    mat = loadmat(
        input_path,
        squeeze_me=False,
        struct_as_record=True,
    )

    csi = mat["csi"]

    if (
        not isinstance(csi, np.ndarray)
        or csi.dtype.names is None
        or "Data" not in csi.dtype.names
    ):
        raise TypeError(
            "Could not find the Data field in the classic MATLAB csi struct."
        )

    if csi["Data"].size != 1:
        raise ValueError(
            "Expected csi to contain exactly one Data field."
        )

    csi["Data"].flat[0] = data

    if mask is not None:
        if "mask" not in mat:
            raise KeyError("Could not find mask in the classic MATLAB file.")
        mat["mask"] = mask

    # scipy metadata entries cannot be passed back to savemat.
    content = {
        key: _replace_none_for_matlab(value)
        for key, value in mat.items()
        if not key.startswith("__")
    }

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        savemat(
            temporary_path,
            content,
            appendmat=False,
            do_compression=True,
            long_field_names=True,
        )
        os.replace(temporary_path, output_path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()

def _replace_none_for_matlab(value):
    """Recursively replace None values by empty MATLAB-compatible arrays."""
    if value is None:
        return np.empty((0, 0), dtype=np.float64)

    if isinstance(value, dict):
        return {
            key: _replace_none_for_matlab(child)
            for key, child in value.items()
        }

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            cleaned = np.empty(value.shape, dtype=object)

            for index in np.ndindex(value.shape):
                cleaned[index] = _replace_none_for_matlab(value[index])

            return cleaned

        if value.dtype.names is not None:
            cleaned = value.copy()

            for field in value.dtype.names:
                field_values = cleaned[field]

                if field_values.dtype == object:
                    for index in np.ndindex(field_values.shape):
                        field_values[index] = _replace_none_for_matlab(
                            field_values[index]
                        )

            return cleaned

    return value
