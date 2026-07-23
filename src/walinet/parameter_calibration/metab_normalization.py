from pathlib import Path
import re

import numpy as np
import yaml

from walinet.config.build import build_config
from walinet.config.build_simulation import build_simulation_config
from walinet.training_data.lcmodel_basis.acquisition import (
    prepare_basis_for_acquisition,
)


def _load_yaml_mapping(path):
    path = Path(path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found:\n  {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)

    if not isinstance(raw, dict):
        raise TypeError(
            "Configuration file must contain a YAML mapping:\n"
            f"  file: {path}\n"
            f"  found: {type(raw)}"
        )

    return raw


def _load_prepared_basis(train_config_path):
    """
    Load only the simulation configuration and prepared LCModel basis.
    No training pool, GPU resources, or simulators are constructed.
    """
    train_config_path = Path(
        train_config_path
    ).expanduser().resolve()

    train_raw = _load_yaml_mapping(
        train_config_path
    )

    train_cfg = build_config(
        train_raw,
        config_dir=train_config_path.parent,
    )

    simulation_config_path = Path(
        train_cfg.data.simulation_config
    ).resolve()

    simulation_raw = _load_yaml_mapping(
        simulation_config_path
    )

    simulation_cfg = build_simulation_config(
        simulation_raw,
        config_dir=simulation_config_path.parent,
    )

    prepared_basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=(
            simulation_cfg.acquisition.bandwidth_hz
        ),
        target_n_timepoints=(
            simulation_cfg.acquisition.n_timepoints
        ),
        dataset_name="clean_fid",
    )

    return prepared_basis


def calculate_r_maps(
    calibration_maps,
    train_config_path,
    batch_size=4096,
):
    """
    Calculate, for every subject and voxel,

        r_i = c_i / max(abs(S_broadened))

    using the voxelwise LCModel FWHM.

    Returns per subject:
        - r_maps
        - coefficients
        - brain_mask
        - fwhm_hz
        - basis_names
        - matched_basis_names
        - unmatched_map_names
        - missing_basis_names
    """

    def key(name):
        name = re.sub(
            r"^\d+[_-]+",
            "",
            str(name),
        )

        return re.sub(
            r"[^a-z0-9]",
            "",
            name.lower(),
        )

    def to_numpy(array):
        if hasattr(array, "detach"):
            array = array.detach().cpu().numpy()

        return np.asarray(array)

    prepared_basis = _load_prepared_basis(
        train_config_path
    )

    basis_fids = to_numpy(
        prepared_basis.fids
    ).astype(
        np.complex64,
        copy=False,
    )

    basis_names = list(
        prepared_basis.names
    )

    if basis_fids.ndim != 2:
        raise ValueError(
            "prepared_basis.fids must have shape "
            "(n_metabolites, n_timepoints), "
            f"but has shape {basis_fids.shape}."
        )

    if basis_fids.shape[0] != len(basis_names):
        raise ValueError(
            "The number of basis FIDs does not match "
            "the number of basis names."
        )

    n_metabolites = len(basis_names)
    n_timepoints = basis_fids.shape[-1]

    dwell_time = float(
        prepared_basis.dwell_time
    )

    hz_per_ppm = float(
        prepared_basis.hz_per_ppm
    )

    time_axis = (
        np.arange(
            n_timepoints,
            dtype=np.float32,
        )
        * dwell_time
    )

    basis_keys = {
        key(name)
        for name in basis_names
    }

    results = {}

    for subject_id, subject in calibration_maps.items():
        raw_metabolite_maps = subject["metabolites"]

        metabolite_maps = {
            key(name): np.asarray(
                values,
                dtype=np.float32,
            )
            for name, values in raw_metabolite_maps.items()
        }

        matched_basis_names = [
            name
            for name in basis_names
            if key(name) in metabolite_maps
        ]

        # Geladen, aber nicht in der Basis gefunden
        unmatched_map_names = [
            name
            for name in raw_metabolite_maps
            if key(name) not in basis_keys
        ]

        # In der Basis, aber keine entsprechende Map geladen
        missing_basis_names = [
            name
            for name in basis_names
            if key(name) not in metabolite_maps
        ]

        if not matched_basis_names:
            raise ValueError(
                f"{subject_id}: no metabolite maps could be matched "
                "to the prepared-basis names."
            )

        fwhm_ppm = np.asarray(
            subject["fwhm"],
            dtype=np.float32,
        )

        spatial_shape = fwhm_ppm.shape

        for name, values in metabolite_maps.items():
            if values.shape != spatial_shape:
                raise ValueError(
                    f"{subject_id}: metabolite map {name} has shape "
                    f"{values.shape}, but the FWHM map has shape "
                    f"{spatial_shape}."
                )

        coefficients = np.stack(
            [
                metabolite_maps.get(
                    key(name),
                    np.zeros(
                        spatial_shape,
                        dtype=np.float32,
                    ),
                )
                for name in basis_names
            ],
            axis=-1,
        )

        # LCModel FWHM: ppm -> Hz
        fwhm_hz = (
            fwhm_ppm
            * hz_per_ppm
        )

        brain_mask = (
            np.all(
                np.isfinite(coefficients),
                axis=-1,
            )
            & np.isfinite(fwhm_hz)
            & (fwhm_hz >= 0)
            & np.any(
                coefficients != 0,
                axis=-1,
            )
        )

        coefficients_flat = coefficients.reshape(
            -1,
            n_metabolites,
        )

        fwhm_hz_flat = fwhm_hz.reshape(-1)

        voxel_indices = np.flatnonzero(
            brain_mask.reshape(-1)
        )

        r_flat = np.full(
            coefficients_flat.shape,
            np.nan,
            dtype=np.float32,
        )

        for start in range(
            0,
            len(voxel_indices),
            batch_size,
        ):
            indices = voxel_indices[
                start:start + batch_size
            ]

            c = coefficients_flat[indices]

            combined_fids = (
                c
                @ basis_fids
            )

            decay = np.exp(
                -np.pi
                * fwhm_hz_flat[indices, None]
                * time_axis[None, :]
            ).astype(
                np.float32,
                copy=False,
            )

            broadened_fids = (
                combined_fids
                * decay
            )

            spectra = np.fft.fft(
                broadened_fids,
                axis=-1,
            )

            spectra = np.fft.fftshift(
                spectra,
                axes=-1,
            )

            maximum = np.max(
                np.abs(spectra),
                axis=-1,
            )

            valid_batch = (
                np.isfinite(maximum)
                & (maximum > 0)
            )

            valid_indices = indices[
                valid_batch
            ]

            r_flat[valid_indices] = (
                c[valid_batch]
                / maximum[valid_batch, None]
            ).astype(
                np.float32,
                copy=False,
            )

        r_maps = r_flat.reshape(
            *spatial_shape,
            n_metabolites,
        )

        results[subject_id] = {
            "r_maps": r_maps,
            "coefficients": coefficients,
            "brain_mask": brain_mask,
            "fwhm_hz": fwhm_hz,
            "basis_names": basis_names,
            "matched_basis_names": matched_basis_names,
            "unmatched_map_names": unmatched_map_names,
            "missing_basis_names": missing_basis_names,
        }

        print(
            f"{subject_id}: "
            f"{int(brain_mask.sum())} valid voxels, "
            f"{len(matched_basis_names)}/{len(raw_metabolite_maps)} "
            "loaded metabolite maps matched"
        )

        if unmatched_map_names:
            print(
                "  Loaded maps not used:",
                unmatched_map_names,
            )

    return results