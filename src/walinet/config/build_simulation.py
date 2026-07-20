# src/walinet/config/build_simulation.py

from __future__ import annotations

import math
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


def _calculate_hz_per_ppm(
    nmr_frequency_hz: float,
) -> float:
    """
    Calculate the frequency difference in Hz corresponding to 1 ppm.

    Example
    -------
    At approximately 7 T proton frequency:

        297222931 Hz / 1e6
        = 297.222931 Hz/ppm
    """
    nmr_frequency_hz = float(
        nmr_frequency_hz
    )

    if (
        not math.isfinite(nmr_frequency_hz)
        or nmr_frequency_hz <= 0
    ):
        raise ValueError(
            "acquisition.nmr_frequency_hz must be finite "
            "and greater than zero."
        )

    return (
        nmr_frequency_hz
        / 1e6
    )


def _ppm_to_hz(
    value_ppm: float,
    *,
    hz_per_ppm: float,
) -> float:
    """
    Convert a frequency value from ppm to Hz.
    """
    value_ppm = float(
        value_ppm
    )

    if not math.isfinite(
        value_ppm
    ):
        raise ValueError(
            "ppm configuration values must be finite."
        )

    return (
        value_ppm
        * hz_per_ppm
    )


def _degrees_to_radians(
    value_deg: float,
) -> float:
    """
    Convert degrees to radians.
    """
    value_deg = float(
        value_deg
    )

    if not math.isfinite(
        value_deg
    ):
        raise ValueError(
            "Phase configuration values in degrees "
            "must be finite."
        )

    return math.radians(
        value_deg
    )


def _degrees_per_ppm_to_radians_per_hz(
    value_deg_per_ppm: float,
    *,
    hz_per_ppm: float,
) -> float:
    """
    Convert a first-order phase slope from deg/ppm to rad/Hz.

    Conversion
    ----------
    deg/ppm
        -> rad/ppm
        -> rad/Hz
    """
    value_deg_per_ppm = float(
        value_deg_per_ppm
    )

    if not math.isfinite(
        value_deg_per_ppm
    ):
        raise ValueError(
            "First-order phase values in deg/ppm "
            "must be finite."
        )

    return (
        math.radians(
            value_deg_per_ppm
        )
        / hz_per_ppm
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
    Build and validate the internal typed simulation configuration.

    The YAML configuration uses human-readable LCModel-style units:

        frequency shift:
            ppm

        FWHM:
            ppm

        zero-order phase:
            degrees

        first-order phase:
            degrees per ppm

    This builder converts these values into the internal units expected
    by the existing simulator:

        frequency shift:
            Hz

        FWHM:
            Hz

        zero-order phase:
            radians

        first-order phase:
            radians per Hz

    The resulting SimulationConfig therefore remains fully compatible
    with the existing simulator code.
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
    acquisition_raw = raw[
        "acquisition"
    ]

    acquisition = AcquisitionCfg(
        bandwidth_hz=float(
            acquisition_raw[
                "bandwidth_hz"
            ]
        ),
        n_timepoints=int(
            acquisition_raw[
                "n_timepoints"
            ]
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
            acquisition_raw[
                "nmr_frequency_hz"
            ]
        ),
    )

    # Conversion factor used for all ppm-based quantities.
    hz_per_ppm = _calculate_hz_per_ppm(
        acquisition.nmr_frequency_hz
    )

    # ---------------------------------------------------------
    # Basis
    # ---------------------------------------------------------
    basis_raw = raw[
        "basis"
    ]

    basis = BasisCfg(
        library=_resolve_path(
            str(
                basis_raw[
                    "library"
                ]
            ),
            config_dir,
        ),
    )

    # ---------------------------------------------------------
    # Metabolite profiles
    # ---------------------------------------------------------
    metabolites_raw = raw[
        "metabolites"
    ]

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

    profiles: list[
        MetaboliteProfileCfg
    ] = []

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
                        profile_raw[
                            "config"
                        ]
                    ),
                    config_dir,
                ),
                probability=float(
                    profile_raw[
                        "probability"
                    ]
                ),
            )
        )

    # ---------------------------------------------------------
    # Frequency shift
    #
    # YAML:
    #     ppm
    #
    # Internal SimulationConfig:
    #     Hz
    # ---------------------------------------------------------
    frequency_shift_raw = metabolites_raw[
        "frequency_shift"
    ]

    frequency_shift = FrequencyShiftCfg(
        mean_hz=_ppm_to_hz(
            frequency_shift_raw[
                "mean_ppm"
            ],
            hz_per_ppm=hz_per_ppm,
        ),
        std_hz=_ppm_to_hz(
            frequency_shift_raw[
                "std_ppm"
            ],
            hz_per_ppm=hz_per_ppm,
        ),
    )

    # ---------------------------------------------------------
    # FWHM
    #
    # YAML:
    #     ppm
    #
    # Internal SimulationConfig:
    #     Hz
    # ---------------------------------------------------------
    fwhm_raw = metabolites_raw[
        "fwhm"
    ]

    fwhm = FWHMCfg(
        mean_hz=_ppm_to_hz(
            fwhm_raw[
                "mean_ppm"
            ],
            hz_per_ppm=hz_per_ppm,
        ),
        std_hz=_ppm_to_hz(
            fwhm_raw[
                "std_ppm"
            ],
            hz_per_ppm=hz_per_ppm,
        ),
    )

    # ---------------------------------------------------------
    # Zero-order phase
    #
    # YAML:
    #     degrees
    #
    # Internal SimulationConfig:
    #     radians
    # ---------------------------------------------------------
    zero_order_phase_raw = metabolites_raw[
        "zero_order_phase"
    ]

    zero_order_phase = ZeroOrderPhaseCfg(
        mean_rad=_degrees_to_radians(
            zero_order_phase_raw[
                "mean_deg"
            ]
        ),
        std_rad=_degrees_to_radians(
            zero_order_phase_raw[
                "std_deg"
            ]
        ),
    )

    # ---------------------------------------------------------
    # First-order phase
    #
    # YAML:
    #     degrees per ppm
    #
    # Internal SimulationConfig:
    #     radians per Hz
    # ---------------------------------------------------------
    first_order_phase_raw = metabolites_raw[
        "first_order_phase"
    ]

    first_order_phase = FirstOrderPhaseCfg(
        mean_rad_per_hz=(
            _degrees_per_ppm_to_radians_per_hz(
                first_order_phase_raw[
                    "mean_deg_per_ppm"
                ],
                hz_per_ppm=hz_per_ppm,
            )
        ),
        std_rad_per_hz=(
            _degrees_per_ppm_to_radians_per_hz(
                first_order_phase_raw[
                    "std_deg_per_ppm"
                ],
                hz_per_ppm=hz_per_ppm,
            )
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
    noise_raw = raw[
        "noise"
    ]

    snr_raw = noise_raw[
        "snr"
    ]

    noise = NoiseCfg(
        snr=SNRDistributionCfg(
            mean=float(
                snr_raw[
                    "mean"
                ]
            ),
            std=float(
                snr_raw[
                    "std"
                ]
            ),
            min=float(
                snr_raw[
                    "min"
                ]
            ),
        )
    )

    # ---------------------------------------------------------
    # Water
    # ---------------------------------------------------------
    water_raw = raw[
        "water"
    ]

    water_scaling_raw = water_raw[
        "scaling"
    ]

    water = WaterCfg(
        scaling=PositiveScalingDistributionCfg(
            mean=float(
                water_scaling_raw[
                    "mean"
                ]
            ),
            std=float(
                water_scaling_raw[
                    "std"
                ]
            ),
        )
    )

    # ---------------------------------------------------------
    # Lipids
    # ---------------------------------------------------------
    lipids_raw = raw[
        "lipids"
    ]

    lipid_scaling_raw = lipids_raw[
        "scaling"
    ]

    lipids = LipidCfg(
        n_random_fids=int(
            lipids_raw[
                "n_random_fids"
            ]
        ),
        scaling=PositiveScalingDistributionCfg(
            mean=float(
                lipid_scaling_raw[
                    "mean"
                ]
            ),
            std=float(
                lipid_scaling_raw[
                    "std"
                ]
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
    # Complete internal runtime configuration
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