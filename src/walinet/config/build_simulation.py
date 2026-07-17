# src/walinet/config/build_simulation.py

from __future__ import annotations

from pathlib import Path

from .schema_simulation import (
    AcquisitionCfg,
    BasisCfg,
    FirstOrderPhaseCfg,
    FrequencyShiftCfg,
    FWHMCfg,
    LipidCfg,
    LipidProjectionCfg,
    MetaboliteCfg,
    MetaboliteProfileCfg,
    NoiseCfg,
    PositiveScalingDistributionCfg,
    SimulationConfig,
    SNRDistributionCfg,
    SubjectSamplingCfg,
    WaterCfg,
    ZeroOrderPhaseCfg,
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

    basis_library = Path(
        cfg.basis.library
    )

    if not basis_library.is_file():
        raise FileNotFoundError(
            "Prepared basis library not found:\n"
            f"  {basis_library}"
        )

    for profile in cfg.metabolites.profiles:
        profile_path = Path(
            profile.config
        )

        if not profile_path.is_file():
            raise FileNotFoundError(
                "Metabolite profile configuration not found:\n"
                f"  {profile_path}"
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
        library=_resolve_path(
            str(
                basis_raw["library"]
            ),
            config_dir,
        ),
    )

    # ---------------------------------------------------------
    # Metabolite profiles
    # ---------------------------------------------------------
    metabolites_raw = raw["metabolites"]

    profiles_raw = metabolites_raw[
        "profiles"
    ]

    if not isinstance(
        profiles_raw,
        list,
    ):
        raise TypeError(
            "metabolites.profiles must be a list."
        )

    profiles: list[MetaboliteProfileCfg] = []

    for profile_index, profile_raw in enumerate(
        profiles_raw
    ):
        if not isinstance(
            profile_raw,
            dict,
        ):
            raise TypeError(
                "Each metabolites.profiles entry must be "
                "a mapping. Invalid entry at index "
                f"{profile_index}."
            )

        profiles.append(
            MetaboliteProfileCfg(
                config=_resolve_path(
                    str(
                        profile_raw["config"]
                    ),
                    config_dir,
                ),
                probability=float(
                    profile_raw["probability"]
                ),
            )
        )

    # ---------------------------------------------------------
    # Remaining metabolite parameters
    # ---------------------------------------------------------
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

    fwhm_raw = metabolites_raw[
        "fwhm"
    ]

    fwhm = FWHMCfg(
        mean_hz=float(
            fwhm_raw["mean_hz"]
        ),
        std_hz=float(
            fwhm_raw["std_hz"]
        ),
    )

    zero_order_phase_raw = metabolites_raw[
        "zero_order_phase"
    ]

    zero_order_phase = ZeroOrderPhaseCfg(
        mean_rad=float(
            zero_order_phase_raw["mean_rad"]
        ),
        std_rad=float(
            zero_order_phase_raw["std_rad"]
        ),
    )

    first_order_phase_raw = metabolites_raw[
        "first_order_phase"
    ]

    first_order_phase = FirstOrderPhaseCfg(
        mean_rad_per_hz=float(
            first_order_phase_raw[
                "mean_rad_per_hz"
            ]
        ),
        std_rad_per_hz=float(
            first_order_phase_raw[
                "std_rad_per_hz"
            ]
        ),
    )

    metabolites = MetaboliteCfg(
        profiles=tuple(
            profiles
        ),
        frequency_shift=frequency_shift,
        fwhm=fwhm,
        zero_order_phase=zero_order_phase,
        first_order_phase=first_order_phase,
    )

    # ---------------------------------------------------------
    # Noise
    # ---------------------------------------------------------
    noise_raw = raw["noise"]
    snr_raw = noise_raw["snr"]

    noise = NoiseCfg(
        snr=SNRDistributionCfg(
            mean=float(
                snr_raw["mean"]
            ),
            std=float(
                snr_raw["std"]
            ),
            min=float(
                snr_raw["min"]
            ),
        )
    )

    # ---------------------------------------------------------
    # Water
    # ---------------------------------------------------------
    water_raw = raw["water"]
    water_scaling_raw = water_raw["scaling"]

    water = WaterCfg(
        scaling=PositiveScalingDistributionCfg(
            mean=float(
                water_scaling_raw["mean"]
            ),
            std=float(
                water_scaling_raw["std"]
            ),
        )
    )

    # ---------------------------------------------------------
    # Lipids
    # ---------------------------------------------------------
    lipids_raw = raw["lipids"]
    lipid_scaling_raw = lipids_raw["scaling"]

    lipids = LipidCfg(
        n_random_fids=int(
            lipids_raw["n_random_fids"]
        ),
        scaling=PositiveScalingDistributionCfg(
            mean=float(
                lipid_scaling_raw["mean"]
            ),
            std=float(
                lipid_scaling_raw["std"]
            ),
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