# src/walinet/config/build_simulation.py

from __future__ import annotations

from pathlib import Path

from .schema_simulation import (
    AcquisitionCfg,
    BasisCfg,
    FrequencyShiftCfg,
    FWHMCfg,
    LipidCfg,
    LipidProjectionCfg,
    MetaboliteCfg,
    NoiseCfg,
    SimulationConfig,
    SubjectSamplingCfg,
    WaterCfg,
)


def _resolve_path(
    path: str,
    config_dir: Path | None,
) -> str:
    """
    Resolve a path relative to the directory containing the
    simulation YAML.

    Empty paths remain empty.
    """
    path = path.strip()

    if not path:
        return ""

    resolved_path = Path(
        path
    ).expanduser()

    if (
        config_dir is not None
        and not resolved_path.is_absolute()
    ):
        resolved_path = (
            config_dir
            / resolved_path
        )

    return str(
        resolved_path.resolve()
    )


def validate_simulation_config(
    cfg: SimulationConfig,
) -> None:
    """
    Validate cross-component configuration constraints.

    Parameter-specific validation is performed by the individual
    frozen dataclasses in schema_simulation.py.
    """
    if not cfg.version.strip():
        raise ValueError(
            "version must not be empty."
        )

    if not cfg.basis.config.strip():
        raise ValueError(
            "basis.config must not be empty."
        )

    if not cfg.metabolites.config.strip():
        raise ValueError(
            "metabolites.config must not be empty."
        )

    # The stored subject-specific projection operator is only
    # unambiguous in the exact legacy-style subject association.
    if (
        cfg.lipid_projection.enabled
        and cfg.subject_sampling.mixing
        != "same_subject"
    ):
        raise ValueError(
            "lipid_projection.enabled may currently only be "
            "used with "
            "subject_sampling.mixing='same_subject'."
        )


def build_simulation_config(
    raw: dict,
    config_dir: Path | None = None,
) -> SimulationConfig:
    """
    Build and validate a typed simulation configuration.

    Paths are resolved relative to the directory containing the
    simulation YAML.
    """

    # ---------------------------------------------------------
    # Version
    # ---------------------------------------------------------
    version = str(
        raw["version"]
    )

    # ---------------------------------------------------------
    # Acquisition
    # ---------------------------------------------------------
    acquisition_raw = raw["acquisition"]

    acquisition = AcquisitionCfg(
        bandwidth_hz=float(
            acquisition_raw["bandwidth_hz"]
        ),
        n_timepoints=int(
            acquisition_raw["n_timepoints"]
        ),
        min_acquired_n_timepoints=int(
            acquisition_raw[
                "min_acquired_n_timepoints"
            ]
        ),
        max_acquired_n_timepoints=int(
            acquisition_raw[
                "max_acquired_n_timepoints"
            ]
        ),
        nmr_frequency_hz=float(
            acquisition_raw["nmr_frequency_hz"]
        ),
    )

    # ---------------------------------------------------------
    # Basis
    # ---------------------------------------------------------
    basis_raw = raw["basis"]

    basis = BasisCfg(
        config=_resolve_path(
            str(
                basis_raw["config"]
            ),
            config_dir,
        ),
    )

    # ---------------------------------------------------------
    # Metabolites
    # ---------------------------------------------------------
    metabolites_raw = raw["metabolites"]

    frequency_shift_raw = metabolites_raw[
        "frequency_shift"
    ]

    frequency_shift = FrequencyShiftCfg(
        mean_hz=float(
            frequency_shift_raw["mean_hz"]
        ),
        std_hz=float(
            frequency_shift_raw["std_hz"]
        ),
    )

    fwhm_raw = metabolites_raw["fwhm"]

    fwhm = FWHMCfg(
        mean_hz=float(
            fwhm_raw["mean_hz"]
        ),
        std_hz=float(
            fwhm_raw["std_hz"]
        ),
    )

    metabolites = MetaboliteCfg(
        config=_resolve_path(
            str(
                metabolites_raw["config"]
            ),
            config_dir,
        ),
        max_acquisition_delay_seconds=float(
            metabolites_raw.get(
                "max_acquisition_delay_seconds",
                0.0,
            )
        ),
        frequency_shift=frequency_shift,
        fwhm=fwhm,
    )

    # ---------------------------------------------------------
    # Noise
    # ---------------------------------------------------------
    noise_raw = raw["noise"]

    noise = NoiseCfg(
        snr_min=float(
            noise_raw["snr_min"]
        ),
        snr_max=float(
            noise_raw["snr_max"]
        ),
    )

    # ---------------------------------------------------------
    # Water
    # ---------------------------------------------------------
    water_raw = raw["water"]

    water = WaterCfg(
        scaling_mean=float(
            water_raw["scaling_mean"]
        ),
        scaling_std=float(
            water_raw["scaling_std"]
        ),
    )

    # ---------------------------------------------------------
    # Lipids
    # ---------------------------------------------------------
    lipids_raw = raw["lipids"]

    lipids = LipidCfg(
        n_random_fids=int(
            lipids_raw["n_random_fids"]
        ),
        scaling_min=float(
            lipids_raw["scaling_min"]
        ),
        scaling_max=float(
            lipids_raw["scaling_max"]
        ),
    )

    # ---------------------------------------------------------
    # Subject sampling
    # ---------------------------------------------------------
    subject_sampling_raw = raw.get(
        "subject_sampling",
        {},
    )

    subject_sampling = SubjectSamplingCfg(
        mixing=str(
            subject_sampling_raw.get(
                "mixing",
                "same_subject",
            )
        ),
    )

    # ---------------------------------------------------------
    # Optional legacy lipid projection
    # ---------------------------------------------------------
    lipid_projection_raw = raw.get(
        "lipid_projection",
        {},
    )

    lipid_projection = LipidProjectionCfg(
        enabled=bool(
            lipid_projection_raw.get(
                "enabled",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Complete configuration
    # ---------------------------------------------------------
    cfg = SimulationConfig(
        version=version,
        acquisition=acquisition,
        basis=basis,
        metabolites=metabolites,
        noise=noise,
        water=water,
        lipids=lipids,
        subject_sampling=subject_sampling,
        lipid_projection=lipid_projection,
    )

    validate_simulation_config(
        cfg
    )

    return cfg