# src/walinet/training_data/metabolite_simulation.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import numpy as np
import torch
import yaml

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.lcmodel_basis.acquisition import (
    PreparedBasis,
)


VOIGT_LORENTZ_COEFFICIENT = 0.5346
VOIGT_LORENTZ_SQUARED_COEFFICIENT = 0.2166


@dataclass(frozen=True)
class MetaboliteSamplingTable:
    """
    One metabolite concentration profile aligned with
    PreparedBasis.names.

    Shapes:
        means:
            (n_basis_components,)

        stds:
            (n_basis_components,)

        enabled_mask:
            (n_basis_components,)
    """

    config_path: str

    basis_names: tuple[str, ...]

    active_config_names: tuple[str, ...]
    active_basis_names: tuple[str, ...]

    means: torch.Tensor
    stds: torch.Tensor
    enabled_mask: torch.Tensor

    @property
    def n_basis_components(self) -> int:
        return len(
            self.basis_names
        )

    @property
    def n_active_components(self) -> int:
        return len(
            self.active_basis_names
        )

    @property
    def device(self) -> torch.device:
        return self.means.device


@dataclass(frozen=True)
class SimulatedMetabolites:
    """
    One batch of noise-free simulated metabolites.

    Shapes:
        clean_fids:
            (batch_size, n_timepoints)

        clean_spectra:
            (batch_size, n_timepoints)

        concentrations:
            (batch_size, n_basis_components)

        profile_indices:
            (batch_size,)

        remaining parameter tensors:
            (batch_size,)
    """

    clean_fids: torch.Tensor
    clean_spectra: torch.Tensor

    concentrations: torch.Tensor
    profile_indices: torch.Tensor

    acquisition_delays_seconds: torch.Tensor
    global_phases_radians: torch.Tensor
    frequency_shifts_hz: torch.Tensor

    voigt_fwhm_hz: torch.Tensor
    lorentzian_fractions: torch.Tensor
    gaussian_fwhm_hz: torch.Tensor
    lorentzian_fwhm_hz: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(
            self.clean_fids.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.clean_fids.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.clean_fids.device


def _load_yaml_mapping(
    path: Path,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            "Metabolite profile configuration not found:\n"
            f"  {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "Metabolite profile configuration must contain "
            "a YAML mapping."
        )

    return raw


def load_metabolite_sampling_table(
    *,
    path: str | Path,
    prepared_basis: PreparedBasis,
    device: torch.device | str,
) -> MetaboliteSamplingTable:
    """
    Load one metabolite profile and align its concentration
    distributions with PreparedBasis.names.

    Every enabled metabolite uses a normal distribution. Negative
    concentration draws are rejected and sampled again later by
    MetaboliteSimulator.

    Disabled or unspecified basis components receive mean=0,
    std=0, and enabled=False.
    """
    path = Path(
        path
    ).resolve()

    device = torch.device(
        device
    )

    raw = _load_yaml_mapping(
        path
    )

    sampling_raw = raw.get(
        "sampling",
        {},
    )

    independent = bool(
        sampling_raw.get(
            "independent",
            True,
        )
    )

    if not independent:
        raise ValueError(
            "Only independent metabolite concentration "
            "sampling is currently supported.\n"
            f"Profile: {path}"
        )

    default_distribution = str(
        sampling_raw.get(
            "default_distribution",
            "normal",
        )
    ).lower()

    if default_distribution != "normal":
        raise ValueError(
            "Only normal concentration distributions "
            "are supported.\n"
            f"Profile: {path}"
        )

    metabolites_raw = raw.get(
        "metabolites"
    )

    if not isinstance(
        metabolites_raw,
        dict,
    ):
        raise TypeError(
            "Metabolite profile must contain a "
            "'metabolites' mapping.\n"
            f"Profile: {path}"
        )

    n_basis_components = (
        prepared_basis.n_metabolites
    )

    means = np.zeros(
        n_basis_components,
        dtype=np.float32,
    )

    stds = np.zeros(
        n_basis_components,
        dtype=np.float32,
    )

    enabled_mask = np.zeros(
        n_basis_components,
        dtype=bool,
    )

    active_config_names: list[str] = []
    active_basis_names: list[str] = []

    used_basis_components: set[str] = set()

    for (
        config_name,
        metabolite_raw,
    ) in metabolites_raw.items():
        if not isinstance(
            metabolite_raw,
            dict,
        ):
            raise TypeError(
                f"Metabolite entry {config_name!r} "
                "must be a mapping.\n"
                f"Profile: {path}"
            )

        enabled = bool(
            metabolite_raw.get(
                "enabled",
                True,
            )
        )

        if not enabled:
            continue

        basis_component_raw = (
            metabolite_raw.get(
                "basis_component"
            )
        )

        if (
            basis_component_raw is None
            or not str(
                basis_component_raw
            ).strip()
        ):
            raise ValueError(
                f"Enabled metabolite {config_name!r} "
                "has no basis_component.\n"
                f"Profile: {path}"
            )

        basis_component = str(
            basis_component_raw
        ).strip()

        if (
            basis_component
            in used_basis_components
        ):
            raise ValueError(
                "Multiple enabled metabolite entries map "
                "to the same basis component:\n"
                f"  {basis_component}\n"
                f"Profile: {path}"
            )

        try:
            basis_index = (
                prepared_basis.index(
                    basis_component
                )
            )

        except KeyError as error:
            raise KeyError(
                f"Metabolite {config_name!r} references "
                "a missing basis component:\n"
                f"  {basis_component}\n"
                f"Profile: {path}"
            ) from error

        distribution_raw = (
            metabolite_raw.get(
                "distribution"
            )
        )

        if not isinstance(
            distribution_raw,
            dict,
        ):
            raise TypeError(
                f"Metabolite {config_name!r} has no "
                "valid distribution mapping.\n"
                f"Profile: {path}"
            )

        distribution_type = str(
            distribution_raw.get(
                "type",
                default_distribution,
            )
        ).lower()

        if distribution_type != "normal":
            raise ValueError(
                f"Metabolite {config_name!r} uses "
                f"unsupported distribution "
                f"{distribution_type!r}.\n"
                f"Profile: {path}"
            )

        mean = float(
            distribution_raw["mean"]
        )

        std = float(
            distribution_raw["std"]
        )

        if not np.isfinite(mean):
            raise ValueError(
                f"Mean for metabolite {config_name!r} "
                "must be finite.\n"
                f"Profile: {path}"
            )

        if (
            not np.isfinite(std)
            or std < 0
        ):
            raise ValueError(
                f"Standard deviation for metabolite "
                f"{config_name!r} must be finite "
                "and >= 0.\n"
                f"Profile: {path}"
            )

        if std == 0 and mean < 0:
            raise ValueError(
                f"Metabolite {config_name!r} has mean < 0 "
                "and std = 0, so a non-negative value can "
                "never be sampled.\n"
                f"Profile: {path}"
            )

        means[basis_index] = mean
        stds[basis_index] = std
        enabled_mask[basis_index] = True

        active_config_names.append(
            str(config_name)
        )

        active_basis_names.append(
            basis_component
        )

        used_basis_components.add(
            basis_component
        )

    if not active_basis_names:
        raise ValueError(
            "No enabled metabolites were found.\n"
            f"Profile: {path}"
        )

    return MetaboliteSamplingTable(
        config_path=str(path),
        basis_names=tuple(
            prepared_basis.names
        ),
        active_config_names=tuple(
            active_config_names
        ),
        active_basis_names=tuple(
            active_basis_names
        ),
        means=torch.from_numpy(
            means
        ).to(
            device=device
        ),
        stds=torch.from_numpy(
            stds
        ).to(
            device=device
        ),
        enabled_mask=torch.from_numpy(
            enabled_mask
        ).to(
            device=device
        ),
    )


class MetaboliteSimulator:
    """
    Vectorized noise-free metabolite simulator.

    Processing order:

        1. Sample one metabolite profile per spectrum.
        2. Sample non-negative concentrations from that profile.
        3. Combine PreparedBasis FIDs.
        4. Sample total Voigt FWHM and line-shape mixture.
        5. Apply global phase and normally distributed frequency shift.
        6. Apply Gaussian/Lorentzian FID broadening in Hz.
        7. Apply acquisition delay through a spectral phase ramp.
        8. Transform to fftshifted spectra.

    The simulator has no internal random state. An explicit
    torch.Generator must be supplied for every simulation call.
    """

    def __init__(
        self,
        *,
        prepared_basis: PreparedBasis,
        config: SimulationConfig,
        device: torch.device | str,
        max_relative_bandwidth_error: float = 1e-3,
    ) -> None:
        self.prepared_basis = (
            prepared_basis
        )

        self.config = config

        self.device = torch.device(
            device
        )

        self._validate_prepared_basis(
            max_relative_bandwidth_error=(
                max_relative_bandwidth_error
            )
        )

        basis_array = np.ascontiguousarray(
            prepared_basis.fids,
            dtype=np.complex64,
        )

        self.basis_fids = torch.from_numpy(
            basis_array
        ).to(
            device=self.device
        ).contiguous()

        self.sampling_tables = tuple(
            load_metabolite_sampling_table(
                path=profile.config,
                prepared_basis=prepared_basis,
                device=self.device,
            )
            for profile in (
                config
                .metabolites
                .profiles
            )
        )

        self.profile_config_paths = tuple(
            table.config_path
            for table in self.sampling_tables
        )

        self.profile_probabilities = torch.tensor(
            [
                profile.probability
                for profile in (
                    config
                    .metabolites
                    .profiles
                )
            ],
            device=self.device,
            dtype=torch.float32,
        ).contiguous()

        self.profile_means = torch.stack(
            [
                table.means
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_stds = torch.stack(
            [
                table.stds
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.profile_enabled_masks = torch.stack(
            [
                table.enabled_mask
                for table in self.sampling_tables
            ],
            dim=0,
        ).contiguous()

        self.time_axis_seconds = (
            torch.arange(
                prepared_basis.n_timepoints,
                device=self.device,
                dtype=torch.float32,
            )
            * float(
                prepared_basis.dwell_time
            )
        )

        self.frequency_axis_hz = (
            torch.fft.fftfreq(
                prepared_basis.n_timepoints,
                d=float(
                    prepared_basis.dwell_time
                ),
                device=self.device,
                dtype=torch.float32,
            )
        )

    @property
    def n_profiles(self) -> int:
        return len(
            self.sampling_tables
        )

    @property
    def n_basis_components(self) -> int:
        return int(
            self.basis_fids.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.basis_fids.shape[-1]
        )

    def simulate(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> SimulatedMetabolites:
        """
        Simulate one batch of noise-free metabolite FIDs and spectra.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        self._validate_generator_device(
            generator
        )

        profile_indices = (
            self._sample_profile_indices(
                batch_size=batch_size,
                generator=generator,
            )
        )

        concentrations = (
            self._sample_concentrations(
                profile_indices=(
                    profile_indices
                ),
                generator=generator,
            )
        )

        metabolite_fids = (
            concentrations.to(
                dtype=self.basis_fids.dtype
            )
            @ self.basis_fids
        )

        acquisition_delays = (
            self._sample_symmetric_uniform(
                maximum=(
                    self.config
                    .metabolites
                    .max_acquisition_delay_seconds
                ),
                batch_size=batch_size,
                generator=generator,
            )
        )

        global_phases = self._sample_uniform(
            minimum=0.0,
            maximum=2.0 * math.pi,
            batch_size=batch_size,
            generator=generator,
        )

        frequency_shift_cfg = (
            self.config
            .metabolites
            .frequency_shift
        )

        frequency_shifts = self._sample_normal(
            mean=frequency_shift_cfg.mean_hz,
            std=frequency_shift_cfg.std_hz,
            shape=(batch_size,),
            generator=generator,
        )

        fwhm_cfg = (
            self.config
            .metabolites
            .fwhm
        )

        voigt_fwhm_hz = (
            self._sample_positive_normal(
                mean=fwhm_cfg.mean_hz,
                std=fwhm_cfg.std_hz,
                batch_size=batch_size,
                generator=generator,
            )
        )

        lorentzian_fractions = (
            self._sample_uniform(
                minimum=0.0,
                maximum=1.0,
                batch_size=batch_size,
                generator=generator,
            )
        )

        (
            lorentzian_fwhm_hz,
            gaussian_fwhm_hz,
        ) = self._split_voigt_fwhm(
            voigt_fwhm_hz=voigt_fwhm_hz,
            lorentzian_fractions=(
                lorentzian_fractions
            ),
        )

        affected_fids = (
            self._apply_fid_effects(
                metabolite_fids=(
                    metabolite_fids
                ),
                global_phases=(
                    global_phases
                ),
                frequency_shifts_hz=(
                    frequency_shifts
                ),
                gaussian_fwhm_hz=(
                    gaussian_fwhm_hz
                ),
                lorentzian_fwhm_hz=(
                    lorentzian_fwhm_hz
                ),
            )
        )

        clean_fids = (
            self._apply_acquisition_delay(
                affected_fids,
                acquisition_delays,
            )
        )

        clean_spectra = (
            torch.fft.fftshift(
                torch.fft.fft(
                    clean_fids,
                    dim=-1,
                ),
                dim=-1,
            )
            .contiguous()
        )

        result = SimulatedMetabolites(
            clean_fids=clean_fids,
            clean_spectra=clean_spectra,
            concentrations=(
                concentrations
            ),
            profile_indices=(
                profile_indices
            ),
            acquisition_delays_seconds=(
                acquisition_delays
            ),
            global_phases_radians=(
                global_phases
            ),
            frequency_shifts_hz=(
                frequency_shifts
            ),
            voigt_fwhm_hz=(
                voigt_fwhm_hz
            ),
            lorentzian_fractions=(
                lorentzian_fractions
            ),
            gaussian_fwhm_hz=(
                gaussian_fwhm_hz
            ),
            lorentzian_fwhm_hz=(
                lorentzian_fwhm_hz
            ),
        )

        self._validate_result(
            result
        )

        return result

    def _sample_profile_indices(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample one metabolite profile index per spectrum.
        """
        if self.n_profiles == 1:
            return torch.zeros(
                (batch_size,),
                device=self.device,
                dtype=torch.long,
            )

        return torch.multinomial(
            self.profile_probabilities,
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        ).contiguous()

    def _sample_concentrations(
        self,
        *,
        profile_indices: torch.Tensor,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample independent non-negative metabolite concentrations
        from the selected profile of each spectrum.

        Negative draws are rejected and sampled again. They are not
        clipped to zero.
        """
        batch_size = int(
            profile_indices.shape[0]
        )

        means = self.profile_means.index_select(
            0,
            profile_indices,
        )

        stds = self.profile_stds.index_select(
            0,
            profile_indices,
        )

        enabled_mask = (
            self.profile_enabled_masks
            .index_select(
                0,
                profile_indices,
            )
        )

        concentrations = (
            means
            + stds
            * torch.randn(
                (
                    batch_size,
                    self.n_basis_components,
                ),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
        )

        invalid = (
            enabled_mask
            & (concentrations < 0)
        )

        while torch.any(invalid):
            invalid_indices = torch.nonzero(
                invalid,
                as_tuple=False,
            )

            batch_indices = (
                invalid_indices[:, 0]
            )

            component_indices = (
                invalid_indices[:, 1]
            )

            redrawn_values = (
                means[
                    batch_indices,
                    component_indices,
                ]
                + stds[
                    batch_indices,
                    component_indices,
                ]
                * torch.randn(
                    (invalid_indices.shape[0],),
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
            )

            concentrations[
                batch_indices,
                component_indices,
            ] = redrawn_values

            invalid = (
                enabled_mask
                & (concentrations < 0)
            )

        concentrations = torch.where(
            enabled_mask,
            concentrations,
            torch.zeros_like(
                concentrations
            ),
        )

        return concentrations.contiguous()

    def _sample_normal(
        self,
        *,
        mean: float,
        std: float,
        shape: tuple[int, ...],
        generator: torch.Generator,
    ) -> torch.Tensor:
        if not math.isfinite(mean):
            raise ValueError(
                "Normal mean must be finite."
            )

        if (
            not math.isfinite(std)
            or std < 0
        ):
            raise ValueError(
                "Normal std must be finite and >= 0."
            )

        if std == 0:
            return torch.full(
                shape,
                fill_value=float(mean),
                device=self.device,
                dtype=torch.float32,
            )

        return (
            float(mean)
            + float(std)
            * torch.randn(
                shape,
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
        )

    def _sample_positive_normal(
        self,
        *,
        mean: float,
        std: float,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        """
        Sample from a normal distribution truncated to values > 0
        using rejection sampling.
        """
        if std == 0 and mean <= 0:
            raise ValueError(
                "A positive normal sample is impossible when "
                "std == 0 and mean <= 0."
            )

        values = self._sample_normal(
            mean=mean,
            std=std,
            shape=(batch_size,),
            generator=generator,
        )

        invalid = values <= 0

        while torch.any(invalid):
            n_invalid = int(
                invalid.sum().item()
            )

            values[invalid] = self._sample_normal(
                mean=mean,
                std=std,
                shape=(n_invalid,),
                generator=generator,
            )

            invalid = values <= 0

        return values.contiguous()

    def _sample_uniform(
        self,
        *,
        minimum: float,
        maximum: float,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if maximum < minimum:
            raise ValueError(
                "maximum must be >= minimum."
            )

        if maximum == minimum:
            return torch.full(
                (batch_size,),
                fill_value=float(
                    minimum
                ),
                device=self.device,
                dtype=torch.float32,
            )

        random_values = torch.rand(
            (batch_size,),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        return (
            minimum
            + (
                maximum
                - minimum
            )
            * random_values
        )

    def _sample_symmetric_uniform(
        self,
        *,
        maximum: float,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        if maximum < 0:
            raise ValueError(
                "maximum must be >= 0."
            )

        return self._sample_uniform(
            minimum=-maximum,
            maximum=maximum,
            batch_size=batch_size,
            generator=generator,
        )

    def _split_voigt_fwhm(
        self,
        *,
        voigt_fwhm_hz: torch.Tensor,
        lorentzian_fractions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Split total Voigt FWHM into Lorentzian and Gaussian FWHM.

        Uses the approximation:

            V = a*L + sqrt(b*L^2 + G^2)

        with:

            a = 0.5346
            b = 0.2166

        lorentzian_fractions is uniform in [0, 1] and interpolates
        between the pure Gaussian and pure Lorentzian endpoints.
        """
        if torch.any(
            voigt_fwhm_hz <= 0
        ):
            raise ValueError(
                "voigt_fwhm_hz must be > 0."
            )

        if torch.any(
            (lorentzian_fractions < 0)
            | (lorentzian_fractions > 1)
        ):
            raise ValueError(
                "lorentzian_fractions must be in [0, 1]."
            )

        a = VOIGT_LORENTZ_COEFFICIENT
        b = VOIGT_LORENTZ_SQUARED_COEFFICIENT

        maximum_lorentzian_fwhm = (
            voigt_fwhm_hz
            / (
                a
                + math.sqrt(b)
            )
        )

        lorentzian_fwhm_hz = (
            lorentzian_fractions
            * maximum_lorentzian_fwhm
        )

        gaussian_fwhm_squared = (
            (
                voigt_fwhm_hz
                - a
                * lorentzian_fwhm_hz
            ).square()
            - b
            * lorentzian_fwhm_hz.square()
        )

        gaussian_fwhm_hz = torch.sqrt(
            torch.clamp(
                gaussian_fwhm_squared,
                min=0.0,
            )
        )

        reconstructed_voigt_fwhm = (
            a
            * lorentzian_fwhm_hz
            + torch.sqrt(
                b
                * lorentzian_fwhm_hz.square()
                + gaussian_fwhm_hz.square()
            )
        )

        if not torch.allclose(
            reconstructed_voigt_fwhm,
            voigt_fwhm_hz,
            rtol=1e-5,
            atol=1e-5,
        ):
            raise RuntimeError(
                "Voigt FWHM decomposition failed its "
                "reconstruction check."
            )

        return (
            lorentzian_fwhm_hz.contiguous(),
            gaussian_fwhm_hz.contiguous(),
        )

    def _apply_acquisition_delay(
        self,
        metabolite_fids: torch.Tensor,
        delays_seconds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply x(t + delay) using a linear phase ramp in the
        unshifted FFT domain.

        For the FFT convention used by PyTorch:

            x(t + tau)
            <->
            X(f) * exp(+i * 2*pi*f*tau)
        """
        if torch.all(
            delays_seconds == 0
        ):
            return metabolite_fids.contiguous()

        spectra = torch.fft.fft(
            metabolite_fids,
            dim=-1,
        )

        phase_angles = (
            2.0
            * math.pi
            * delays_seconds[:, None]
            * self.frequency_axis_hz[
                None,
                :
            ]
        )

        phase_ramp = torch.polar(
            torch.ones_like(
                phase_angles
            ),
            phase_angles,
        )

        delayed_fids = torch.fft.ifft(
            spectra
            * phase_ramp,
            dim=-1,
        )

        return delayed_fids.contiguous()

    def _apply_fid_effects(
        self,
        *,
        metabolite_fids: torch.Tensor,
        global_phases: torch.Tensor,
        frequency_shifts_hz: torch.Tensor,
        gaussian_fwhm_hz: torch.Tensor,
        lorentzian_fwhm_hz: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply phase, frequency shift and Voigt broadening.

        Lorentzian FID decay:

            exp(-pi * L * t)

        Gaussian FID decay:

            exp(-(pi * G * t)^2 / (4*ln(2)))

        where L and G are the Lorentzian and Gaussian FWHM in Hz.
        """
        time_axis = (
            self.time_axis_seconds[
                None,
                :
            ]
        )

        phase_angles = (
            global_phases[:, None]
            + 2.0
            * math.pi
            * frequency_shifts_hz[:, None]
            * time_axis
        )

        phase_factor = torch.polar(
            torch.ones_like(
                phase_angles
            ),
            phase_angles,
        )

        lorentzian_exponent = (
            math.pi
            * lorentzian_fwhm_hz[:, None]
            * time_axis
        )

        gaussian_exponent = (
            (
                math.pi
                * gaussian_fwhm_hz[:, None]
                * time_axis
            ).square()
            / (
                4.0
                * math.log(2.0)
            )
        )

        decay_factor = torch.exp(
            -lorentzian_exponent
            -gaussian_exponent
        )

        result = (
            metabolite_fids
            * phase_factor
            * decay_factor
        )

        return result.contiguous()

    def _validate_prepared_basis(
        self,
        *,
        max_relative_bandwidth_error: float,
    ) -> None:
        basis = self.prepared_basis

        if basis.fids.ndim != 2:
            raise ValueError(
                "PreparedBasis.fids must have shape "
                "(n_metabolites, n_timepoints)."
            )

        if (
            len(basis.names)
            != basis.fids.shape[0]
        ):
            raise ValueError(
                "PreparedBasis.names does not match "
                "PreparedBasis.fids."
            )

        if (
            basis.n_timepoints
            != basis.fids.shape[-1]
        ):
            raise ValueError(
                "PreparedBasis.n_timepoints does not "
                "match PreparedBasis.fids."
            )

        expected_n_timepoints = (
            self.config
            .acquisition
            .n_timepoints
        )

        if (
            basis.n_timepoints
            != expected_n_timepoints
        ):
            raise ValueError(
                "Prepared basis and simulation configuration "
                "use different FID lengths:\n"
                f"  basis:      {basis.n_timepoints}\n"
                f"  simulation: {expected_n_timepoints}"
            )

        if not np.all(
            np.isfinite(
                basis.fids
            )
        ):
            raise ValueError(
                "PreparedBasis.fids contains NaN or Inf."
            )

        expected_bandwidth = (
            self.config
            .acquisition
            .bandwidth_hz
        )

        relative_error = abs(
            basis.bandwidth
            - expected_bandwidth
        ) / expected_bandwidth

        if (
            relative_error
            > max_relative_bandwidth_error
        ):
            raise ValueError(
                "Prepared basis bandwidth differs too strongly "
                "from the simulation bandwidth:\n"
                f"  basis:          {basis.bandwidth}\n"
                f"  simulation:     {expected_bandwidth}\n"
                f"  relative error: {relative_error:.3e}\n"
                f"  maximum:        "
                f"{max_relative_bandwidth_error:.3e}"
            )

    def _validate_generator_device(
        self,
        generator: torch.Generator,
    ) -> None:
        generator_device = torch.device(
            generator.device
        )

        if (
            generator_device.type
            != self.device.type
        ):
            raise ValueError(
                "Generator and simulator must use "
                "the same device type:\n"
                f"  generator: {generator_device}\n"
                f"  simulator: {self.device}"
            )

        if (
            self.device.type == "cuda"
            and generator_device.index is not None
            and self.device.index is not None
            and generator_device.index
            != self.device.index
        ):
            raise ValueError(
                "Generator and simulator must use "
                "the same CUDA device."
            )

    def _validate_result(
        self,
        result: SimulatedMetabolites,
    ) -> None:
        expected_signal_shape = (
            result.batch_size,
            self.n_timepoints,
        )

        if (
            result.clean_fids.shape
            != expected_signal_shape
        ):
            raise RuntimeError(
                "Unexpected clean_fids shape."
            )

        if (
            result.clean_spectra.shape
            != expected_signal_shape
        ):
            raise RuntimeError(
                "Unexpected clean_spectra shape."
            )

        expected_concentration_shape = (
            result.batch_size,
            self.n_basis_components,
        )

        if (
            result.concentrations.shape
            != expected_concentration_shape
        ):
            raise RuntimeError(
                "Unexpected concentrations shape."
            )

        if tuple(
            result.profile_indices.shape
        ) != (
            result.batch_size,
        ):
            raise RuntimeError(
                "Unexpected profile_indices shape."
            )

        if result.profile_indices.dtype != torch.long:
            raise RuntimeError(
                "profile_indices must use torch.long dtype."
            )

        if torch.any(
            result.profile_indices < 0
        ):
            raise RuntimeError(
                "profile_indices contains negative values."
            )

        if torch.any(
            result.profile_indices >= self.n_profiles
        ):
            raise RuntimeError(
                "profile_indices contains an out-of-range value."
            )

        parameter_tensors = {
            "acquisition_delays_seconds": (
                result.acquisition_delays_seconds
            ),
            "global_phases_radians": (
                result.global_phases_radians
            ),
            "frequency_shifts_hz": (
                result.frequency_shifts_hz
            ),
            "voigt_fwhm_hz": (
                result.voigt_fwhm_hz
            ),
            "lorentzian_fractions": (
                result.lorentzian_fractions
            ),
            "gaussian_fwhm_hz": (
                result.gaussian_fwhm_hz
            ),
            "lorentzian_fwhm_hz": (
                result.lorentzian_fwhm_hz
            ),
        }

        for name, tensor in parameter_tensors.items():
            if tuple(tensor.shape) != (
                result.batch_size,
            ):
                raise RuntimeError(
                    f"Unexpected shape for {name}: "
                    f"{tuple(tensor.shape)}"
                )

            if not torch.isfinite(tensor).all():
                raise RuntimeError(
                    f"{name} contains non-finite values."
                )

        if not torch.isfinite(
            result.concentrations
        ).all():
            raise RuntimeError(
                "Metabolite concentrations contain "
                "non-finite values."
            )

        if torch.any(
            result.concentrations < 0
        ):
            raise RuntimeError(
                "Metabolite concentrations contain negative values."
            )

        selected_enabled_masks = (
            self.profile_enabled_masks
            .index_select(
                0,
                result.profile_indices,
            )
        )

        if torch.any(
            result.concentrations[
                ~selected_enabled_masks
            ]
            != 0
        ):
            raise RuntimeError(
                "Disabled metabolites contain non-zero "
                "concentrations."
            )

        if torch.any(
            result.voigt_fwhm_hz <= 0
        ):
            raise RuntimeError(
                "Voigt FWHM contains non-positive values."
            )

        if torch.any(
            result.gaussian_fwhm_hz < 0
        ):
            raise RuntimeError(
                "Gaussian FWHM contains negative values."
            )

        if torch.any(
            result.lorentzian_fwhm_hz < 0
        ):
            raise RuntimeError(
                "Lorentzian FWHM contains negative values."
            )

        if not torch.isfinite(
            result.clean_fids.real
        ).all():
            raise RuntimeError(
                "Metabolite FIDs contain non-finite "
                "real values."
            )

        if not torch.isfinite(
            result.clean_fids.imag
        ).all():
            raise RuntimeError(
                "Metabolite FIDs contain non-finite "
                "imaginary values."
            )

        if not torch.isfinite(
            result.clean_spectra.real
        ).all():
            raise RuntimeError(
                "Metabolite spectra contain non-finite "
                "real values."
            )

        if not torch.isfinite(
            result.clean_spectra.imag
        ).all():
            raise RuntimeError(
                "Metabolite spectra contain non-finite "
                "imaginary values."
            )