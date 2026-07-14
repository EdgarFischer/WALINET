# src/walinet/training_data/water_lipid_extraction.py

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from walinet.config.schema_water_lipid import (
    WaterLipidExtractionConfig,
)
from walinet.training_data.water_removal import (
    suppress_water_volume,
)


RESOURCE_FORMAT = "walinet_water_lipid_resources"
RESOURCE_FORMAT_VERSION = "3.0"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_subject_paths(
    cfg: WaterLipidExtractionConfig,
    subject: str,
) -> dict[str, Path]:
    """
    Resolve input and output paths for one subject.
    """
    subject_dir = Path(cfg.data.base_dir) / subject
    output_dir = subject_dir / cfg.data.paths.output_dir

    resources_filename = (
        cfg.resources.simulation_resources_filename.format(
            version=cfg.version,
        )
    )

    return {
        "subject_dir": subject_dir,
        "brain_mask": subject_dir / cfg.data.paths.brain_mask,
        "lipid_mask": subject_dir / cfg.data.paths.lipid_mask,
        "input_data": subject_dir / cfg.data.paths.input_data,
        "output_dir": output_dir,
        "simulation_resources": output_dir / resources_filename,
    }


def _require_file(
    path: Path,
    *,
    description: str,
) -> None:
    """
    Raise an informative error if a required file is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(
            f"{description} not found: {path}"
        )


def load_subject_data(
    paths: dict[str, Path],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load masks and the complete native FID volume.

    Nothing is cropped along the time dimension.

    Returns:
        brain_mask:
            Boolean array with shape (X, Y, Z).

        lipid_mask:
            Boolean array with shape (X, Y, Z).

        csi_fids:
            Complex array with shape (X, Y, Z, T).
    """
    _require_file(
        paths["brain_mask"],
        description="Brain mask",
    )
    _require_file(
        paths["lipid_mask"],
        description="Lipid mask",
    )
    _require_file(
        paths["input_data"],
        description="Original FID data",
    )

    brain_mask_raw = np.load(
        paths["brain_mask"],
        allow_pickle=False,
    )
    lipid_mask_raw = np.load(
        paths["lipid_mask"],
        allow_pickle=False,
    )
    csi_fids = np.load(
        paths["input_data"],
        allow_pickle=False,
    )

    if csi_fids.ndim != 4:
        raise ValueError(
            "Original data must have shape "
            "(X, Y, Z, T), but found "
            f"{csi_fids.shape}."
        )

    spatial_shape = csi_fids.shape[:-1]

    if brain_mask_raw.shape != spatial_shape:
        raise ValueError(
            "Brain-mask shape does not match "
            "the spatial data shape:\n"
            f"  brain mask: {brain_mask_raw.shape}\n"
            f"  data:       {spatial_shape}"
        )

    if lipid_mask_raw.shape != spatial_shape:
        raise ValueError(
            "Lipid-mask shape does not match "
            "the spatial data shape:\n"
            f"  lipid mask: {lipid_mask_raw.shape}\n"
            f"  data:       {spatial_shape}"
        )

    if not np.all(np.isfinite(csi_fids)):
        raise ValueError(
            "Original FID data contains NaN or Inf values."
        )

    if not np.all(np.isfinite(brain_mask_raw)):
        raise ValueError(
            "Brain mask contains NaN or Inf values."
        )

    if not np.all(np.isfinite(lipid_mask_raw)):
        raise ValueError(
            "Lipid mask contains NaN or Inf values."
        )

    brain_mask = brain_mask_raw > 0
    lipid_mask = lipid_mask_raw > 0

    if not np.any(brain_mask):
        raise ValueError(
            "Brain mask is empty."
        )

    if not np.any(lipid_mask):
        raise ValueError(
            "Lipid mask is empty."
        )

    return (
        brain_mask,
        lipid_mask,
        csi_fids,
    )


def _validate_isolated_water(
    isolated_water: np.ndarray,
    *,
    expected_shape: tuple[int, ...],
) -> np.ndarray:
    """
    Validate the temporary isolated-water volume.

    Shape:
        (X, Y, Z, T)
    """
    isolated_water = np.asarray(
        isolated_water,
        dtype=np.complex64,
    )

    if isolated_water.shape != expected_shape:
        raise ValueError(
            "Isolated-water shape does not match "
            "the original data:\n"
            f"  water: {isolated_water.shape}\n"
            f"  data:  {expected_shape}"
        )

    if not np.all(np.isfinite(isolated_water)):
        raise ValueError(
            "Isolated water contains NaN or Inf values."
        )

    return isolated_water


def _valid_fid_rows(
    fids: np.ndarray,
) -> np.ndarray:
    """
    Return one boolean value per FID.

    The final dimension is interpreted as the time dimension.

    A valid FID:
        - contains only finite values;
        - is not identically zero.
    """
    finite = np.isfinite(
        fids
    ).all(
        axis=-1,
    )

    nonempty = np.any(
        fids != 0,
        axis=-1,
    )

    return finite & nonempty


def compute_isolated_water(
    *,
    subject: str,
    cfg: WaterLipidExtractionConfig,
    csi_fids: np.ndarray,
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
) -> np.ndarray:
    """
    Extract water using HSVD.

    HSVD is applied inside:

        brain_mask | lipid_mask

    Water in the lipid mask is required temporarily so that it can
    be subtracted from the original lipid-mask FIDs.

    The returned array has shape (X, Y, Z, T), but it is not saved
    directly. It exists only in memory during preprocessing.
    """
    print(
        f"[Water] Computing isolated water for {subject}..."
    )

    head_mask = np.logical_or(
        brain_mask,
        lipid_mask,
    )

    isolated_water = suppress_water_volume(
        image_grid=np.asarray(csi_fids),
        mask=head_mask,
        cfg=cfg.water_extraction,
    )

    isolated_water = _validate_isolated_water(
        isolated_water,
        expected_shape=csi_fids.shape,
    )

    # Keep the complete spatial shape, but force everything
    # outside brain_mask | lipid_mask to zero.
    isolated_water = np.where(
        head_mask[..., None],
        isolated_water,
        0.0,
    ).astype(
        np.complex64,
        copy=False,
    )

    isolated_water = np.ascontiguousarray(
        isolated_water,
        dtype=np.complex64,
    )

    print("[Water] Isolated-water calculation finished.")
    print(f"  Shape: {isolated_water.shape}")
    print(f"  Dtype: {isolated_water.dtype}")
    print("  Domain: FID")
    print("  Temporary mask: brain_mask | lipid_mask")
    print("  The temporary full water volume is not saved.")

    return isolated_water


def extract_simulation_resources(
    *,
    csi_fids: np.ndarray,
    isolated_water: np.ndarray,
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the final FID-domain simulation resources.

    Water:
        Complete spatial volume with shape (X, Y, Z, T).

        Only voxels inside the brain mask contain isolated-water
        FIDs. Everything outside the brain mask is zero.

    Lipids:
        Compact pool with shape (N_valid_lipid_voxels, T).

        Isolated water is subtracted from each lipid-mask voxel
        before the FID is added to the lipid pool.
    """
    if isolated_water.shape != csi_fids.shape:
        raise ValueError(
            "Isolated-water shape does not match "
            "the original-data shape:\n"
            f"  water: {isolated_water.shape}\n"
            f"  data:  {csi_fids.shape}"
        )

    spatial_shape = csi_fids.shape[:-1]

    if brain_mask.shape != spatial_shape:
        raise ValueError(
            "Brain-mask shape does not match "
            "the original-data shape:\n"
            f"  brain mask: {brain_mask.shape}\n"
            f"  data:       {spatial_shape}"
        )

    if lipid_mask.shape != spatial_shape:
        raise ValueError(
            "Lipid-mask shape does not match "
            "the original-data shape:\n"
            f"  lipid mask: {lipid_mask.shape}\n"
            f"  data:       {spatial_shape}"
        )

    # -----------------------------------------------------------------
    # Water resource
    # -----------------------------------------------------------------
    # Preserve the complete spatial volume, but retain water only
    # inside the brain mask. Water from the lipid mask is no longer
    # needed after the lipid pool has been calculated.
    water_fids = np.zeros(
        csi_fids.shape,
        dtype=np.complex64,
    )

    water_fids[brain_mask] = np.asarray(
        isolated_water[brain_mask],
        dtype=np.complex64,
    )

    water_fids = np.ascontiguousarray(
        water_fids,
        dtype=np.complex64,
    )

    if not np.all(np.isfinite(water_fids)):
        raise ValueError(
            "Water FIDs contain NaN or Inf values."
        )

    if np.any(water_fids[~brain_mask] != 0):
        raise RuntimeError(
            "Water FIDs outside the brain mask "
            "are unexpectedly non-zero."
        )

    brain_water_fids = water_fids[
        brain_mask
    ]

    valid_brain_water = _valid_fid_rows(
        brain_water_fids
    )

    n_valid_water = int(
        valid_brain_water.sum()
    )
    n_empty_water = int(
        (~valid_brain_water).sum()
    )

    if n_valid_water == 0:
        raise ValueError(
            "No valid water FIDs were found "
            "inside the brain mask."
        )

    if n_empty_water > 0:
        print(
            f"[Water] Found {n_empty_water} empty "
            "water FIDs inside the brain mask."
        )
        print(
            "[Water] Their spatial positions remain "
            "stored as zero FIDs."
        )

    # -----------------------------------------------------------------
    # Lipid resource
    # -----------------------------------------------------------------
    # Water was also extracted in the lipid mask specifically so that
    # it can now be removed from the original lipid-mask signal.
    lipid_fids = (
        np.asarray(
            csi_fids[lipid_mask],
            dtype=np.complex64,
        )
        - np.asarray(
            isolated_water[lipid_mask],
            dtype=np.complex64,
        )
    )

    valid_lipid_fids = _valid_fid_rows(
        lipid_fids
    )

    n_invalid_lipids = int(
        (~valid_lipid_fids).sum()
    )

    if n_invalid_lipids > 0:
        print(
            f"[Lipids] Removing {n_invalid_lipids} "
            "invalid or empty lipid FIDs."
        )

    lipid_fids = np.ascontiguousarray(
        lipid_fids[
            valid_lipid_fids
        ],
        dtype=np.complex64,
    )

    if lipid_fids.shape[0] == 0:
        raise ValueError(
            "No valid lipid FIDs remain."
        )

    if not np.all(np.isfinite(lipid_fids)):
        raise ValueError(
            "Lipid FIDs contain NaN or Inf values."
        )

    return (
        water_fids,
        lipid_fids,
    )


def save_simulation_resources(
    *,
    path: Path,
    subject: str,
    cfg: WaterLipidExtractionConfig,
    source_paths: dict[str, Path],
    water_fids: np.ndarray,
    lipid_fids: np.ndarray,
    brain_mask: np.ndarray,
    lipid_mask: np.ndarray,
) -> None:
    """
    Save all final preprocessing results in one HDF5 file.

    Datasets:
        water_fids:
            Shape (X, Y, Z, T).
            Complete spatial structure.
            Water only inside the brain mask.
            Zero outside the brain mask.

        lipid_fids:
            Shape (N_valid_lipid_voxels, T).
            Compact pool.
            Isolated water has already been removed.

        brain_mask:
            Shape (X, Y, Z).
            Stored so the spatial water resource is self-contained.
    """
    if water_fids.ndim != 4:
        raise ValueError(
            "water_fids must have shape "
            "(X, Y, Z, T), but found "
            f"{water_fids.shape}."
        )

    if lipid_fids.ndim != 2:
        raise ValueError(
            "lipid_fids must have shape "
            "(N_lipid_voxels, T), but found "
            f"{lipid_fids.shape}."
        )

    if water_fids.shape[:-1] != brain_mask.shape:
        raise ValueError(
            "Water spatial shape does not match "
            "the brain-mask shape:\n"
            f"  water:      {water_fids.shape[:-1]}\n"
            f"  brain mask: {brain_mask.shape}"
        )

    if water_fids.shape[:-1] != lipid_mask.shape:
        raise ValueError(
            "Water spatial shape does not match "
            "the lipid-mask shape:\n"
            f"  water:      {water_fids.shape[:-1]}\n"
            f"  lipid mask: {lipid_mask.shape}"
        )

    if water_fids.shape[-1] != lipid_fids.shape[-1]:
        raise ValueError(
            "Water and lipid FIDs have different "
            "numbers of timepoints:\n"
            f"  water: {water_fids.shape[-1]}\n"
            f"  lipid: {lipid_fids.shape[-1]}"
        )

    if not np.all(np.isfinite(water_fids)):
        raise ValueError(
            "Water FIDs contain NaN or Inf values."
        )

    if not np.all(np.isfinite(lipid_fids)):
        raise ValueError(
            "Lipid FIDs contain NaN or Inf values."
        )

    if np.any(water_fids[~brain_mask] != 0):
        raise ValueError(
            "Water FIDs outside the brain mask must be zero."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_name(
        f"{path.name}.tmp"
    )

    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with h5py.File(
            temporary_path,
            "w",
        ) as h5:
            h5.attrs["format"] = RESOURCE_FORMAT
            h5.attrs["format_version"] = RESOURCE_FORMAT_VERSION
            h5.attrs["created_utc"] = _utc_now()

            h5.attrs["subject"] = subject
            h5.attrs["preprocessing_version"] = cfg.version

            h5.attrs["bandwidth_hz"] = float(
                cfg.water_extraction.bandwidth
            )
            h5.attrs["dwell_time_seconds"] = float(
                cfg.water_extraction.dwell_time
            )
            h5.attrs["native_n_timepoints"] = int(
                water_fids.shape[-1]
            )

            h5.attrs["domain"] = "fid"
            h5.attrs["fft_shifted"] = False
            h5.attrs["dtype"] = "complex64"

            h5.attrs["water_layout"] = "spatial_volume"
            h5.attrs["water_mask"] = "brain_mask"
            h5.attrs["water_outside_mask_zero"] = True

            h5.attrs["lipid_layout"] = "fid_pool"
            h5.attrs["lipid_water_removed"] = True

            h5.attrs["water_extraction_mask"] = (
                "brain_mask | lipid_mask"
            )
            h5.attrs["temporary_isolated_water_saved"] = False

            h5.attrs["spatial_shape"] = np.asarray(
                water_fids.shape[:-1],
                dtype=np.int64,
            )

            h5.attrs["n_brain_voxels"] = int(
                brain_mask.sum()
            )
            h5.attrs["n_lipid_mask_voxels"] = int(
                lipid_mask.sum()
            )
            h5.attrs["n_saved_lipid_fids"] = int(
                lipid_fids.shape[0]
            )

            h5.attrs["hsvd_components"] = int(
                cfg.water_extraction.hsvd_components
            )
            h5.attrs["water_min_freq_hz"] = float(
                cfg.water_extraction.min_freq
            )
            h5.attrs["water_max_freq_hz"] = float(
                cfg.water_extraction.max_freq
            )

            h5.attrs["source_data_path"] = str(
                source_paths["input_data"]
            )
            h5.attrs["brain_mask_path"] = str(
                source_paths["brain_mask"]
            )
            h5.attrs["lipid_mask_path"] = str(
                source_paths["lipid_mask"]
            )

            h5.create_dataset(
                "water_fids",
                data=np.asarray(
                    water_fids,
                    dtype=np.complex64,
                ),
                compression="lzf",
                shuffle=True,
            )

            h5.create_dataset(
                "lipid_fids",
                data=np.asarray(
                    lipid_fids,
                    dtype=np.complex64,
                ),
                compression="lzf",
                shuffle=True,
            )

            h5.create_dataset(
                "brain_mask",
                data=np.asarray(
                    brain_mask,
                    dtype=np.uint8,
                ),
                compression="lzf",
                shuffle=True,
            )

        # Replace the final file only after the temporary HDF5 file
        # has been written and closed successfully.
        temporary_path.replace(
            path
        )

    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()

        raise

    print("[Resources] Saved:")
    print(f"  {path}")
    print(f"  water_fids: {water_fids.shape}")
    print(f"  lipid_fids: {lipid_fids.shape}")
    print(f"  brain_mask: {brain_mask.shape}")
    print("  Domain: FID")
    print("  No separate isolated-water file was created.")


def process_subject(
    *,
    subject: str,
    cfg: WaterLipidExtractionConfig,
) -> Path:
    """
    Extract and save water/lipid resources for one subject.
    """
    print()
    print("=" * 72)
    print(f"Processing subject: {subject}")
    print("=" * 72)

    paths = get_subject_paths(
        cfg,
        subject,
    )

    resource_path = paths[
        "simulation_resources"
    ]

    if (
        resource_path.is_file()
        and not cfg.resources.overwrite
    ):
        print(
            "[Skip] Simulation resources already exist:"
        )
        print(f"  {resource_path}")

        return resource_path

    if (
        resource_path.exists()
        and cfg.resources.overwrite
    ):
        print(
            "[Resources] Existing file will be overwritten:"
        )
        print(f"  {resource_path}")

    (
        brain_mask,
        lipid_mask,
        csi_fids,
    ) = load_subject_data(
        paths
    )

    print(
        f"Original data shape: {csi_fids.shape}"
    )
    print(
        f"Original data dtype: {csi_fids.dtype}"
    )
    print(
        f"Brain voxels: {int(brain_mask.sum())}"
    )
    print(
        f"Lipid-mask voxels: {int(lipid_mask.sum())}"
    )

    # The complete isolated-water volume exists only temporarily
    # during this subject's preprocessing.
    isolated_water = compute_isolated_water(
        subject=subject,
        cfg=cfg,
        csi_fids=csi_fids,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
    )

    (
        water_fids,
        lipid_fids,
    ) = extract_simulation_resources(
        csi_fids=csi_fids,
        isolated_water=isolated_water,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
    )

    # Free the large temporary water array before writing HDF5.
    del isolated_water

    print(
        f"Water FIDs: {water_fids.shape}"
    )
    print(
        f"Lipid FIDs: {lipid_fids.shape}"
    )

    save_simulation_resources(
        path=resource_path,
        subject=subject,
        cfg=cfg,
        source_paths=paths,
        water_fids=water_fids,
        lipid_fids=lipid_fids,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
    )

    return resource_path


def process_all_subjects(
    cfg: WaterLipidExtractionConfig,
) -> list[Path]:
    """
    Process all subjects listed in the extraction configuration.
    """
    output_paths: list[Path] = []

    n_subjects = len(
        cfg.data.subjects
    )

    for index, subject in enumerate(
        cfg.data.subjects,
        start=1,
    ):
        print()
        print(
            f"Subject {index}/{n_subjects}"
        )

        output_path = process_subject(
            subject=subject,
            cfg=cfg,
        )

        output_paths.append(
            output_path
        )

    print()
    print("=" * 72)
    print(
        f"Finished {n_subjects} subject(s)."
    )
    print("=" * 72)

    return output_paths