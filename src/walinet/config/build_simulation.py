# src/walinet/config/build_simulation.py

from __future__ import annotations

from pathlib import Path

from .schema_simulation import (
    AcquisitionCfg,
    BasisCfg,
    LineBroadeningCfg,
    LipidCfg,
    LipidProjectionCfg,
    MetaboliteCfg,
    NoiseCfg,
    SimulationConfig,
    SubjectSamplingCfg,
    WaterCfg,
)


VALID_LIPID_SCALING_DISTRIBUTIONS = {
    "uniform",
    "log_uniform",
}

VALID_SUBJECT_MIXING_MODES = {
    "same_subject",
    "separate_water_lipid_subjects",
    "independent_lipid_fids",
}


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
    Validate the complete simulation configuration.
    """

    # ---------------------------------------------------------
    # General
    # ---------------------------------------------------------
    if not cfg.version.strip():
        raise ValueError(
            "version must not be empty."
        )

    # ---------------------------------------------------------
    # Acquisition
    # ---------------------------------------------------------
    if cfg.acquisition.bandwidth_hz <= 0:
        raise ValueError(
            "acquisition.bandwidth_hz must be > 0."
        )

    if cfg.acquisition.n_timepoints <= 0:
        raise ValueError(
            "acquisition.n_timepoints must be > 0."
        )

    if (
        cfg.acquisition.min_acquired_n_timepoints
        <= 0
    ):
        raise ValueError(
            "acquisition.min_acquired_n_timepoints "
            "must be > 0."
        )

    if (
        cfg.acquisition.max_acquired_n_timepoints
        < cfg.acquisition.min_acquired_n_timepoints
    ):
        raise ValueError(
            "acquisition.max_acquired_n_timepoints "
            "must be >= "
            "acquisition.min_acquired_n_timepoints."
        )

    if (
        cfg.acquisition.max_acquired_n_timepoints
        > cfg.acquisition.n_timepoints
    ):
        raise ValueError(
            "acquisition.max_acquired_n_timepoints "
            "must be <= acquisition.n_timepoints."
        )

    if cfg.acquisition.nmr_frequency_hz <= 0:
        raise ValueError(
            "acquisition.nmr_frequency_hz must be > 0."
        )

    # ---------------------------------------------------------
    # Basis
    # ---------------------------------------------------------
    if not cfg.basis.config.strip():
        raise ValueError(
            "basis.config must not be empty."
        )

    # ---------------------------------------------------------
    # Metabolites
    # ---------------------------------------------------------
    if not cfg.metabolites.config.strip():
        raise ValueError(
            "metabolites.config must not be empty."
        )

    if (
        cfg.metabolites.max_acquisition_delay_seconds
        < 0
    ):
        raise ValueError(
            "metabolites.max_acquisition_delay_seconds "
            "must be >= 0."
        )

    if (
        cfg.metabolites.max_frequency_shift_hz
        < 0
    ):
        raise ValueError(
            "metabolites.max_frequency_shift_hz "
            "must be >= 0."
        )

    line_broadening = (
        cfg.metabolites.line_broadening
    )

    if line_broadening.minimum < 0:
        raise ValueError(
            "metabolites.line_broadening.min "
            "must be >= 0."
        )

    if (
        line_broadening.maximum
        < line_broadening.minimum
    ):
        raise ValueError(
            "metabolites.line_broadening.max must be "
            ">= metabolites.line_broadening.min."
        )

    if not (
        0.0
        <= line_broadening.gaussian_fraction_min
        <= 1.0
    ):
        raise ValueError(
            "metabolites.line_broadening."
            "gaussian_fraction_min must be in [0, 1]."
        )

    if not (
        0.0
        <= line_broadening.gaussian_fraction_max
        <= 1.0
    ):
        raise ValueError(
            "metabolites.line_broadening."
            "gaussian_fraction_max must be in [0, 1]."
        )

    if (
        line_broadening.gaussian_fraction_max
        < line_broadening.gaussian_fraction_min
    ):
        raise ValueError(
            "metabolites.line_broadening."
            "gaussian_fraction_max must be >= "
            "gaussian_fraction_min."
        )

    # ---------------------------------------------------------
    # Noise
    # ---------------------------------------------------------
    if cfg.noise.snr_min <= 0:
        raise ValueError(
            "noise.snr_min must be > 0."
        )

    if cfg.noise.snr_max < cfg.noise.snr_min:
        raise ValueError(
            "noise.snr_max must be >= noise.snr_min."
        )

    # ---------------------------------------------------------
    # Water
    # ---------------------------------------------------------
    if cfg.water.scaling_min < 0:
        raise ValueError(
            "water.scaling_min must be >= 0."
        )

    if (
        cfg.water.scaling_max
        < cfg.water.scaling_min
    ):
        raise ValueError(
            "water.scaling_max must be >= "
            "water.scaling_min."
        )

    # ---------------------------------------------------------
    # Lipids
    # ---------------------------------------------------------
    if cfg.lipids.n_random_fids <= 0:
        raise ValueError(
            "lipids.n_random_fids must be > 0."
        )

    if cfg.lipids.scaling_min < 0:
        raise ValueError(
            "lipids.scaling_min must be >= 0."
        )

    if (
        cfg.lipids.scaling_max
        < cfg.lipids.scaling_min
    ):
        raise ValueError(
            "lipids.scaling_max must be >= "
            "lipids.scaling_min."
        )

    if (
        cfg.lipids.scaling_distribution
        not in VALID_LIPID_SCALING_DISTRIBUTIONS
    ):
        raise ValueError(
            "lipids.scaling_distribution must be one of "
            f"{sorted(VALID_LIPID_SCALING_DISTRIBUTIONS)}, "
            "but found "
            f"{cfg.lipids.scaling_distribution!r}."
        )

    if (
        cfg.lipids.scaling_distribution
        == "log_uniform"
        and cfg.lipids.scaling_min <= 0
    ):
        raise ValueError(
            "lipids.scaling_min must be > 0 when "
            "lipids.scaling_distribution is "
            "'log_uniform'."
        )

    # ---------------------------------------------------------
    # Subject sampling
    # ---------------------------------------------------------
    if (
        cfg.subject_sampling.mixing
        not in VALID_SUBJECT_MIXING_MODES
    ):
        raise ValueError(
            "subject_sampling.mixing must be one of "
            f"{sorted(VALID_SUBJECT_MIXING_MODES)}, "
            "but found "
            f"{cfg.subject_sampling.mixing!r}."
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
            str(basis_raw["config"]),
            config_dir,
        ),
    )

    # ---------------------------------------------------------
    # Metabolites
    # ---------------------------------------------------------
    metabolites_raw = raw["metabolites"]

    line_broadening_raw = metabolites_raw[
        "line_broadening"
    ]

    line_broadening = LineBroadeningCfg(
        minimum=float(
            line_broadening_raw["min"]
        ),
        maximum=float(
            line_broadening_raw["max"]
        ),
        gaussian_fraction_min=float(
            line_broadening_raw.get(
                "gaussian_fraction_min",
                0.0,
            )
        ),
        gaussian_fraction_max=float(
            line_broadening_raw.get(
                "gaussian_fraction_max",
                1.0,
            )
        ),
    )

    metabolites = MetaboliteCfg(
        config=_resolve_path(
            str(metabolites_raw["config"]),
            config_dir,
        ),
        max_acquisition_delay_seconds=float(
            metabolites_raw.get(
                "max_acquisition_delay_seconds",
                0.0,
            )
        ),
        max_frequency_shift_hz=float(
            metabolites_raw.get(
                "max_frequency_shift_hz",
                0.0,
            )
        ),
        line_broadening=line_broadening,
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
        scaling_min=float(
            water_raw["scaling_min"]
        ),
        scaling_max=float(
            water_raw["scaling_max"]
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
        scaling_distribution=str(
            lipids_raw.get(
                "scaling_distribution",
                "log_uniform",
            )
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