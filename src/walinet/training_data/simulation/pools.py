from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json

import h5py
import numpy as np

from walinet.config.schema_training_data import (
    TrainingDataConfig,
)
from walinet.training_data.water_removal import (
    get_or_create_isolated_water,
    get_subject_paths,
)


SIMULATION_POOL_FORMAT = (
    "walinet_simulation_pools"
)
SIMULATION_POOL_VERSION = "1.0"


@dataclass
class SimulationPools:
    """
    Global water and lipid pools assembled from all training subjects.

    Both pools are stored in the frequency domain with shape

        (n_pool_spectra, n_timepoints)

    and use complex64 for efficient sampling and GPU transfer.
    """

    water_spectra: np.ndarray
    lipid_spectra: np.ndarray

    water_subject_indices: np.ndarray
    lipid_subject_indices: np.ndarray

    subject_names: list[str]

    n_timepoints: int
    bandwidth: float

    pool_path: Path | None = None

    @property
    def n_water_spectra(self) -> int:
        return self.water_spectra.shape[0]

    @property
    def n_lipid_spectra(self) -> int:
        return self.lipid_spectra.shape[0]

    def validate(self) -> None:
        """Validate shapes, dtypes and numerical contents."""
        for name, spectra in (
            (
                "water_spectra",
                self.water_spectra,
            ),
            (
                "lipid_spectra",
                self.lipid_spectra,
            ),
        ):
            if spectra.ndim != 2:
                raise ValueError(
                    f"{name} must have shape "
                    "(n_spectra, n_timepoints), "
                    f"but found {spectra.shape}."
                )

            if (
                spectra.shape[1]
                != self.n_timepoints
            ):
                raise ValueError(
                    f"{name} has "
                    f"{spectra.shape[1]} points; "
                    f"expected {self.n_timepoints}."
                )

            if not np.all(
                np.isfinite(spectra)
            ):
                raise ValueError(
                    f"{name} contains "
                    "non-finite values."
                )

        if (
            self.water_subject_indices.shape
            != (self.n_water_spectra,)
        ):
            raise ValueError(
                "water_subject_indices has "
                "an incompatible shape."
            )

        if (
            self.lipid_subject_indices.shape
            != (self.n_lipid_spectra,)
        ):
            raise ValueError(
                "lipid_subject_indices has "
                "an incompatible shape."
            )

        if self.n_water_spectra == 0:
            raise ValueError(
                "The water pool is empty."
            )

        if self.n_lipid_spectra == 0:
            raise ValueError(
                "The lipid pool is empty."
            )


def _utc_now() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def _fids_to_spectra(
    fids: np.ndarray,
) -> np.ndarray:
    """
    Convert a batch of FIDs to fftshifted frequency spectra.
    """
    fids = np.asarray(
        fids,
        dtype=np.complex64,
    )

    if fids.ndim != 2:
        raise ValueError(
            "FID pool must have shape "
            "(n_spectra, n_timepoints)."
        )

    spectra = np.fft.fftshift(
        np.fft.fft(
            fids,
            axis=-1,
        ),
        axes=-1,
    )

    return np.ascontiguousarray(
        spectra,
        dtype=np.complex64,
    )


def _valid_spectrum_rows(
    spectra: np.ndarray,
    *,
    minimum_maximum: float = 1e-12,
) -> np.ndarray:
    """
    Select finite, non-empty spectra.

    Empty spectra would later cause divisions by zero during random
    water or lipid scaling.
    """
    if spectra.shape[0] == 0:
        return np.zeros(
            0,
            dtype=bool,
        )

    finite = np.isfinite(
        spectra
    ).all(
        axis=-1,
    )

    nonzero = np.max(
        np.abs(spectra),
        axis=-1,
    ) > minimum_maximum

    return finite & nonzero


def _extract_subject_pools(
    *,
    subject: str,
    cfg: TrainingDataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build frequency-domain water and lipid pools for one subject.

    Water is taken from the cached isolated-water volume inside the
    brain mask.

    Lipid signals are taken from

        original FID - isolated water FID

    inside the lipid mask, matching the previous simulator.
    """
    paths = get_subject_paths(
        cfg,
        subject,
    )

    brain_mask = (
        np.load(
            paths["brain_mask"]
        )
        > 0
    )

    lipid_mask = (
        np.load(
            paths["lipid_mask"]
        )
        > 0
    )

    n_timepoints = (
        cfg.acquisition.n_timepoints
    )

    csi_rrrt = np.load(
        paths["data"],
        mmap_mode="r",
    )[..., :n_timepoints]

    # Loads the cached file when available. It only runs the expensive
    # water-removal procedure when no cached result exists.
    water_rrrt = (
        get_or_create_isolated_water(
            subject,
            cfg,
        )[..., :n_timepoints]
    )

    if (
        csi_rrrt.shape
        != water_rrrt.shape
    ):
        raise ValueError(
            f"Original data and isolated "
            f"water have different shapes for "
            f"{subject}:\n"
            f"  original: {csi_rrrt.shape}\n"
            f"  water:    {water_rrrt.shape}"
        )

    if (
        brain_mask.shape
        != csi_rrrt.shape[:-1]
    ):
        raise ValueError(
            f"Brain-mask shape mismatch for "
            f"{subject}."
        )

    if (
        lipid_mask.shape
        != csi_rrrt.shape[:-1]
    ):
        raise ValueError(
            f"Lipid-mask shape mismatch for "
            f"{subject}."
        )

    # Water pool: isolated water in brain voxels.
    water_fids = np.asarray(
        water_rrrt[brain_mask],
        dtype=np.complex64,
    )

    # Lipid pool: water-suppressed signal in lipid-mask voxels.
    # Index first to avoid allocating a complete temporary 4D volume.
    lipid_fids = (
        np.asarray(
            csi_rrrt[lipid_mask],
            dtype=np.complex64,
        )
        - np.asarray(
            water_rrrt[lipid_mask],
            dtype=np.complex64,
        )
    )

    water_spectra = _fids_to_spectra(
        water_fids
    )

    lipid_spectra = _fids_to_spectra(
        lipid_fids
    )

    water_valid = _valid_spectrum_rows(
        water_spectra
    )

    lipid_valid = _valid_spectrum_rows(
        lipid_spectra
    )

    n_water_removed = int(
        (~water_valid).sum()
    )

    n_lipid_removed = int(
        (~lipid_valid).sum()
    )

    if n_water_removed:
        print(
            f"[Pool] {subject}: removed "
            f"{n_water_removed} invalid or "
            "empty water spectra."
        )

    if n_lipid_removed:
        print(
            f"[Pool] {subject}: removed "
            f"{n_lipid_removed} invalid or "
            "empty lipid spectra."
        )

    water_spectra = (
        water_spectra[water_valid]
    )

    lipid_spectra = (
        lipid_spectra[lipid_valid]
    )

    print(
        f"[Pool] {subject}: "
        f"{water_spectra.shape[0]} water, "
        f"{lipid_spectra.shape[0]} lipid"
    )

    return (
        water_spectra,
        lipid_spectra,
    )


def build_simulation_pools(
    cfg: TrainingDataConfig,
) -> SimulationPools:
    """
    Build global water and lipid pools from all configured subjects.

    Only subjects listed in ``cfg.data.subjects`` are included.
    Therefore training, validation and test configurations must use
    separate subject lists.
    """
    subject_names = list(
        cfg.data.subjects
    )

    if not subject_names:
        raise ValueError(
            "No subjects configured for "
            "simulation-pool creation."
        )

    water_parts: list[np.ndarray] = []
    lipid_parts: list[np.ndarray] = []

    water_subject_parts: list[
        np.ndarray
    ] = []

    lipid_subject_parts: list[
        np.ndarray
    ] = []

    for subject_index, subject in enumerate(
        subject_names
    ):
        print()
        print(
            f"Building pools for "
            f"{subject}..."
        )

        water_spectra, lipid_spectra = (
            _extract_subject_pools(
                subject=subject,
                cfg=cfg,
            )
        )

        water_parts.append(
            water_spectra
        )

        lipid_parts.append(
            lipid_spectra
        )

        water_subject_parts.append(
            np.full(
                water_spectra.shape[0],
                subject_index,
                dtype=np.int32,
            )
        )

        lipid_subject_parts.append(
            np.full(
                lipid_spectra.shape[0],
                subject_index,
                dtype=np.int32,
            )
        )

    water_pool = np.ascontiguousarray(
        np.concatenate(
            water_parts,
            axis=0,
        ),
        dtype=np.complex64,
    )

    lipid_pool = np.ascontiguousarray(
        np.concatenate(
            lipid_parts,
            axis=0,
        ),
        dtype=np.complex64,
    )

    water_subject_indices = (
        np.concatenate(
            water_subject_parts,
            axis=0,
        )
    )

    lipid_subject_indices = (
        np.concatenate(
            lipid_subject_parts,
            axis=0,
        )
    )

    pools = SimulationPools(
        water_spectra=water_pool,
        lipid_spectra=lipid_pool,
        water_subject_indices=(
            water_subject_indices
        ),
        lipid_subject_indices=(
            lipid_subject_indices
        ),
        subject_names=subject_names,
        n_timepoints=(
            cfg.acquisition.n_timepoints
        ),
        bandwidth=float(
            cfg.acquisition.bandwidth
        ),
    )

    pools.validate()

    print()
    print("Global simulation pools built")
    print(
        f"  Subjects       : "
        f"{len(subject_names)}"
    )
    print(
        f"  Water spectra  : "
        f"{pools.n_water_spectra}"
    )
    print(
        f"  Lipid spectra  : "
        f"{pools.n_lipid_spectra}"
    )
    print(
        f"  Spectral points: "
        f"{pools.n_timepoints}"
    )
    print(
        f"  Water memory   : "
        f"{pools.water_spectra.nbytes / 1024**2:.1f} MB"
    )
    print(
        f"  Lipid memory   : "
        f"{pools.lipid_spectra.nbytes / 1024**2:.1f} MB"
    )

    return pools


def save_simulation_pools(
    pools: SimulationPools,
    path: str | Path,
) -> None:
    """Save global simulation pools to one HDF5 file."""
    pools.validate()

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(
        path,
        "w",
    ) as h5:
        h5.attrs["format"] = (
            SIMULATION_POOL_FORMAT
        )

        h5.attrs["format_version"] = (
            SIMULATION_POOL_VERSION
        )

        h5.attrs["created_utc"] = (
            _utc_now()
        )

        h5.attrs["n_timepoints"] = (
            pools.n_timepoints
        )

        h5.attrs["bandwidth"] = (
            pools.bandwidth
        )

        h5.attrs["subject_names"] = (
            json.dumps(
                pools.subject_names
            )
        )

        h5.create_dataset(
            "water_spectra",
            data=pools.water_spectra,
            compression="lzf",
            shuffle=True,
        )

        h5.create_dataset(
            "lipid_spectra",
            data=pools.lipid_spectra,
            compression="lzf",
            shuffle=True,
        )

        h5.create_dataset(
            "water_subject_indices",
            data=pools.water_subject_indices,
            compression="lzf",
            shuffle=True,
        )

        h5.create_dataset(
            "lipid_subject_indices",
            data=pools.lipid_subject_indices,
            compression="lzf",
            shuffle=True,
        )

    pools.pool_path = path.resolve()

    print(
        f"Simulation pools saved: "
        f"{path.resolve()}"
    )


def load_simulation_pools(
    path: str | Path,
) -> SimulationPools:
    """Load previously generated simulation pools."""
    path = Path(path).resolve()

    if not path.is_file():
        raise FileNotFoundError(
            f"Simulation-pool file "
            f"does not exist: {path}"
        )

    with h5py.File(
        path,
        "r",
    ) as h5:
        stored_format = h5.attrs.get(
            "format"
        )

        if (
            stored_format
            != SIMULATION_POOL_FORMAT
        ):
            raise ValueError(
                "Not a compatible WALINET "
                "simulation-pool file."
            )

        stored_version = str(
            h5.attrs.get(
                "format_version"
            )
        )

        if (
            stored_version
            != SIMULATION_POOL_VERSION
        ):
            raise ValueError(
                "Unsupported simulation-pool "
                f"version: {stored_version}"
            )

        pools = SimulationPools(
            water_spectra=np.ascontiguousarray(
                h5["water_spectra"][...],
                dtype=np.complex64,
            ),
            lipid_spectra=np.ascontiguousarray(
                h5["lipid_spectra"][...],
                dtype=np.complex64,
            ),
            water_subject_indices=(
                h5[
                    "water_subject_indices"
                ][...].astype(
                    np.int32,
                    copy=False,
                )
            ),
            lipid_subject_indices=(
                h5[
                    "lipid_subject_indices"
                ][...].astype(
                    np.int32,
                    copy=False,
                )
            ),
            subject_names=json.loads(
                h5.attrs[
                    "subject_names"
                ]
            ),
            n_timepoints=int(
                h5.attrs[
                    "n_timepoints"
                ]
            ),
            bandwidth=float(
                h5.attrs[
                    "bandwidth"
                ]
            ),
            pool_path=path,
        )

    pools.validate()

    print(
        f"Simulation pools loaded: "
        f"{path}"
    )
    print(
        f"  Water spectra: "
        f"{pools.n_water_spectra}"
    )
    print(
        f"  Lipid spectra: "
        f"{pools.n_lipid_spectra}"
    )

    return pools


def get_or_create_simulation_pools(
    *,
    cfg: TrainingDataConfig,
    path: str | Path,
    overwrite: bool = False,
) -> SimulationPools:
    """
    Load existing global pools or create and save them when absent.
    """
    path = Path(path)

    if path.exists() and not overwrite:
        return load_simulation_pools(
            path
        )

    if path.exists() and overwrite:
        print(
            f"Existing simulation pools "
            f"will be overwritten: {path}"
        )

    pools = build_simulation_pools(
        cfg
    )

    save_simulation_pools(
        pools,
        path,
    )

    return pools