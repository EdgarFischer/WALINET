# src/walinet/config/schema_simulation.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AcquisitionCfg:
    """
    Acquisition parameters of the simulated spectra.
    """

    bandwidth_hz: float
    n_timepoints: int
    nmr_frequency_hz: float

    @property
    def dwell_time_seconds(self) -> float:
        """
        Time between consecutive FID samples.
        """
        return 1.0 / self.bandwidth_hz


@dataclass(frozen=True)
class BasisCfg:
    """
    Path to the LCModel-basis preparation configuration.
    """

    config: str


@dataclass(frozen=True)
class LineBroadeningCfg:
    """
    Parameters of the legacy mixed Gaussian/Lorentzian
    FID damping.

    For every simulated spectrum:

        total ~ Uniform(minimum, maximum)

        gaussian_fraction ~ Uniform(
            gaussian_fraction_min,
            gaussian_fraction_max,
        )

        gaussian_coefficient =
            gaussian_fraction * total

        lorentzian_coefficient =
            (1 - gaussian_fraction) * total
    """

    minimum: float
    maximum: float
    gaussian_fraction_min: float
    gaussian_fraction_max: float


@dataclass(frozen=True)
class MetaboliteCfg:
    """
    Metabolite simulation parameters.

    Concentration distributions and basis-component mappings are
    stored in the separate metabolite configuration.
    """

    config: str
    max_acquisition_delay_seconds: float
    max_frequency_shift_hz: float
    line_broadening: LineBroadeningCfg


@dataclass(frozen=True)
class NoiseCfg:
    """
    Range from which the target SNR is sampled uniformly.
    """

    snr_min: float
    snr_max: float


@dataclass(frozen=True)
class WaterCfg:
    """
    Water scaling relative to the metabolite-spectrum amplitude.
    """

    scaling_min: float
    scaling_max: float


@dataclass(frozen=True)
class LipidCfg:
    """
    Lipid baseline simulation parameters.
    """

    n_random_fids: int
    scaling_min: float
    scaling_max: float
    scaling_distribution: str


@dataclass(frozen=True)
class SubjectSamplingCfg:
    """
    Controls how water and lipid resources are associated with
    subjects.

    Supported modes:

        same_subject:
            Water and all lipid FIDs come from the same subject.

        separate_water_lipid_subjects:
            Water comes from one subject and all lipid FIDs come
            from one independently selected subject.

        independent_lipid_fids:
            Each individual lipid FID may come from a different
            subject.
    """

    mixing: str


@dataclass(frozen=True)
class LipidProjectionCfg:
    """
    Optional legacy lipid-projection output.

    When enabled, the subject-specific stored operator is applied
    to the complete simulated spectrum and the projected spectrum
    is returned in addition to the normal simulator output.
    """

    enabled: bool


@dataclass(frozen=True)
class SimulationConfig:
    version: str
    acquisition: AcquisitionCfg
    basis: BasisCfg
    metabolites: MetaboliteCfg
    noise: NoiseCfg
    water: WaterCfg
    lipids: LipidCfg
    subject_sampling: SubjectSamplingCfg
    lipid_projection: LipidProjectionCfg