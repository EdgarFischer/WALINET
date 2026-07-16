# src/walinet/config/schema_simulation.py

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AcquisitionCfg:
    """
    Acquisition parameters of the simulated spectra.

    n_timepoints:
        Fixed internal FID length and final spectral length.

    min_acquired_n_timepoints:
        Minimum number of actually acquired FID samples.

    max_acquired_n_timepoints:
        Maximum number of actually acquired FID samples.

    Samples after the selected acquisition length are set to zero
    before the final FFT.
    """

    bandwidth_hz: float
    n_timepoints: int

    min_acquired_n_timepoints: int
    max_acquired_n_timepoints: int

    nmr_frequency_hz: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.bandwidth_hz)
            or self.bandwidth_hz <= 0
        ):
            raise ValueError(
                "acquisition.bandwidth_hz must be "
                "finite and > 0."
            )

        if self.n_timepoints <= 0:
            raise ValueError(
                "acquisition.n_timepoints must be > 0."
            )

        if self.min_acquired_n_timepoints <= 0:
            raise ValueError(
                "acquisition.min_acquired_n_timepoints "
                "must be > 0."
            )

        if (
            self.max_acquired_n_timepoints
            < self.min_acquired_n_timepoints
        ):
            raise ValueError(
                "acquisition.max_acquired_n_timepoints "
                "must be >= "
                "acquisition.min_acquired_n_timepoints."
            )

        if (
            self.max_acquired_n_timepoints
            > self.n_timepoints
        ):
            raise ValueError(
                "acquisition.max_acquired_n_timepoints "
                "must be <= acquisition.n_timepoints."
            )

        if (
            not math.isfinite(self.nmr_frequency_hz)
            or self.nmr_frequency_hz <= 0
        ):
            raise ValueError(
                "acquisition.nmr_frequency_hz must be "
                "finite and > 0."
            )

    @property
    def dwell_time_seconds(self) -> float:
        """
        Time between consecutive FID samples.
        """
        return 1.0 / self.bandwidth_hz


@dataclass(frozen=True)
class BasisCfg:
    """
    Path to the prepared LCModel basis library.
    """

    library: str

    def __post_init__(self) -> None:
        if not self.library.strip():
            raise ValueError(
                "basis.library must not be empty."
            )


@dataclass(frozen=True)
class MetaboliteProfileCfg:
    """
    One metabolite concentration profile.

    config:
        Path to the profile-specific Metabos YAML.

    probability:
        Probability of selecting this profile for one complete
        simulated spectrum.
    """

    config: str
    probability: float

    def __post_init__(self) -> None:
        if not self.config.strip():
            raise ValueError(
                "metabolites.profiles[].config must not be empty."
            )

        if (
            not math.isfinite(self.probability)
            or self.probability <= 0
        ):
            raise ValueError(
                "metabolites.profiles[].probability must be "
                "finite and > 0."
            )


@dataclass(frozen=True)
class FrequencyShiftCfg:
    """
    Normal distribution of the global frequency shift in Hz.

    Negative and positive values are allowed.
    """

    mean_hz: float
    std_hz: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.mean_hz):
            raise ValueError(
                "metabolites.frequency_shift.mean_hz "
                "must be finite."
            )

        if (
            not math.isfinite(self.std_hz)
            or self.std_hz < 0
        ):
            raise ValueError(
                "metabolites.frequency_shift.std_hz "
                "must be finite and >= 0."
            )


@dataclass(frozen=True)
class FWHMCfg:
    """
    Normal distribution of the total Voigt FWHM in Hz.

    Non-positive draws are rejected and sampled again.
    """

    mean_hz: float
    std_hz: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.mean_hz)
            or self.mean_hz <= 0
        ):
            raise ValueError(
                "metabolites.fwhm.mean_hz "
                "must be finite and > 0."
            )

        if (
            not math.isfinite(self.std_hz)
            or self.std_hz < 0
        ):
            raise ValueError(
                "metabolites.fwhm.std_hz "
                "must be finite and >= 0."
            )


@dataclass(frozen=True)
class MetaboliteCfg:
    """
    Metabolite simulation parameters.

    One profile is sampled for each complete simulated spectrum.
    All metabolite concentrations of that spectrum are then drawn
    from the selected profile.
    """

    profiles: tuple[MetaboliteProfileCfg, ...]

    max_acquisition_delay_seconds: float

    frequency_shift: FrequencyShiftCfg
    fwhm: FWHMCfg

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ValueError(
                "metabolites.profiles must contain at least "
                "one profile."
            )

        probability_sum = sum(
            profile.probability
            for profile in self.profiles
        )

        if not math.isclose(
            probability_sum,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "metabolites.profiles probabilities must sum "
                "to 1. Found: "
                f"{probability_sum:.12g}"
            )

        if (
            not math.isfinite(
                self.max_acquisition_delay_seconds
            )
            or self.max_acquisition_delay_seconds < 0
        ):
            raise ValueError(
                "metabolites.max_acquisition_delay_seconds "
                "must be finite and >= 0."
            )


@dataclass(frozen=True)
class NoiseCfg:
    """
    Range from which the target SNR is sampled uniformly.
    """

    snr_min: float
    snr_max: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.snr_min)
            or self.snr_min <= 0
        ):
            raise ValueError(
                "noise.snr_min must be finite and > 0."
            )

        if not math.isfinite(self.snr_max):
            raise ValueError(
                "noise.snr_max must be finite."
            )

        if self.snr_max < self.snr_min:
            raise ValueError(
                "noise.snr_max must be >= noise.snr_min."
            )


@dataclass(frozen=True)
class WaterCfg:
    """
    Normal distribution of water scaling relative to the maximum
    absolute metabolite-spectrum amplitude.

    Non-positive draws are rejected and sampled again.
    """

    scaling_mean: float
    scaling_std: float

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.scaling_mean)
            or self.scaling_mean <= 0
        ):
            raise ValueError(
                "water.scaling_mean must be "
                "finite and > 0."
            )

        if (
            not math.isfinite(self.scaling_std)
            or self.scaling_std < 0
        ):
            raise ValueError(
                "water.scaling_std must be "
                "finite and >= 0."
            )


@dataclass(frozen=True)
class LipidCfg:
    """
    Lipid baseline simulation parameters.

    Lipid scaling is always sampled log-uniformly.
    """

    n_random_fids: int
    scaling_min: float
    scaling_max: float

    def __post_init__(self) -> None:
        if self.n_random_fids <= 0:
            raise ValueError(
                "lipids.n_random_fids must be > 0."
            )

        if (
            not math.isfinite(self.scaling_min)
            or self.scaling_min <= 0
        ):
            raise ValueError(
                "lipids.scaling_min must be "
                "finite and > 0."
            )

        if not math.isfinite(self.scaling_max):
            raise ValueError(
                "lipids.scaling_max must be finite."
            )

        if self.scaling_max < self.scaling_min:
            raise ValueError(
                "lipids.scaling_max must be >= "
                "lipids.scaling_min."
            )


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

    def __post_init__(self) -> None:
        supported_modes = {
            "same_subject",
            "separate_water_lipid_subjects",
            "independent_lipid_fids",
        }

        if self.mixing not in supported_modes:
            raise ValueError(
                "Unsupported subject_sampling.mixing: "
                f"{self.mixing!r}. Supported modes: "
                f"{sorted(supported_modes)}"
            )


@dataclass(frozen=True)
class LipidProjectionCfg:
    """
    Optional legacy lipid-projection output.

    When enabled, the subject-specific stored operator is applied
    to the complete simulated spectrum.
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