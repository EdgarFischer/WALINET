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


@dataclass(frozen=True)
class MetaboliteSamplingTable:
    """
    Concentration distributions aligned with PreparedBasis.names.

    Shapes:
        means:
            (n_basis_components,)

        stds:
            (n_basis_components,)

        enabled_mask:
            (n_basis_components,)
    """

    basis_names: tuple[str, ...]

    active_config_names: tuple[str, ...]
    active_basis_names: tuple[str, ...]

    means: torch.Tensor
    stds: torch.Tensor
    enabled_mask: torch.Tensor

    clip_negative_values: bool
    minimum_concentration: float

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

        remaining parameter tensors:
            (batch_size,)
    """

    clean_fids: torch.Tensor
    clean_spectra: torch.Tensor

    concentrations: torch.Tensor

    acquisition_delays_seconds: torch.Tensor
    global_phases_radians: torch.Tensor
    frequency_shifts_hz: torch.Tensor

    total_broadening: torch.Tensor
    gaussian_fractions: torch.Tensor
    gaussian_broadening: torch.Tensor
    lorentzian_broadening: torch.Tensor

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
            "Metabolite configuration not found:\n"
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
            "Metabolite configuration must contain "
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
    Load Metabos.yaml and align its concentration distributions
    with PreparedBasis.names.

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
            "sampling is currently supported."
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
            "are currently supported."
        )

    clip_negative_values = bool(
        sampling_raw.get(
            "clip_negative_values",
            True,
        )
    )

    minimum_concentration = float(
        sampling_raw.get(
            "minimum_concentration",
            0.0,
        )
    )

    if not np.isfinite(
        minimum_concentration
    ):
        raise ValueError(
            "sampling.minimum_concentration "
            "must be finite."
        )

    if minimum_concentration < 0:
        raise ValueError(
            "sampling.minimum_concentration "
            "must be >= 0."
        )

    metabolites_raw = raw.get(
        "metabolites"
    )

    if not isinstance(
        metabolites_raw,
        dict,
    ):
        raise TypeError(
            "Metabos.yaml must contain a "
            "'metabolites' mapping."
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
                "must be a mapping."
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
                "has no basis_component."
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
                f"  {basis_component}"
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
                f"  {basis_component}"
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
                "valid distribution mapping."
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
                f"{distribution_type!r}."
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
                "must be finite."
            )

        if (
            not np.isfinite(std)
            or std < 0
        ):
            raise ValueError(
                f"Standard deviation for metabolite "
                f"{config_name!r} must be finite "
                "and >= 0."
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
            "No enabled metabolites were found."
        )

    table = MetaboliteSamplingTable(
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
        clip_negative_values=(
            clip_negative_values
        ),
        minimum_concentration=(
            minimum_concentration
        ),
    )

    return table


class MetaboliteSimulator:
    """
    Vectorized noise-free metabolite simulator.

    Processing order:

        1. Sample concentrations.
        2. Combine PreparedBasis FIDs.
        3. Apply acquisition delay through a spectral phase ramp.
        4. Apply global phase.
        5. Apply frequency shift.
        6. Apply Gaussian/Lorentzian FID broadening.
        7. Transform to fftshifted spectra.

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

        self.sampling_table = (
            load_metabolite_sampling_table(
                path=(
                    config
                    .metabolites
                    .config
                ),
                prepared_basis=(
                    prepared_basis
                ),
                device=self.device,
            )
        )

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

        concentrations = (
            self._sample_concentrations(
                batch_size=batch_size,
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

        metabolite_fids = (
            self._apply_acquisition_delay(
                metabolite_fids,
                acquisition_delays,
            )
        )

        global_phases = self._sample_uniform(
            minimum=0.0,
            maximum=2.0 * math.pi,
            batch_size=batch_size,
            generator=generator,
        )

        frequency_shifts = (
            self._sample_symmetric_uniform(
                maximum=(
                    self.config
                    .metabolites
                    .max_frequency_shift_hz
                ),
                batch_size=batch_size,
                generator=generator,
            )
        )

        line_broadening_cfg = (
            self.config
            .metabolites
            .line_broadening
        )

        total_broadening = (
            self._sample_uniform(
                minimum=(
                    line_broadening_cfg
                    .minimum
                ),
                maximum=(
                    line_broadening_cfg
                    .maximum
                ),
                batch_size=batch_size,
                generator=generator,
            )
        )

        gaussian_fractions = (
            self._sample_uniform(
                minimum=(
                    line_broadening_cfg
                    .gaussian_fraction_min
                ),
                maximum=(
                    line_broadening_cfg
                    .gaussian_fraction_max
                ),
                batch_size=batch_size,
                generator=generator,
            )
        )

        gaussian_broadening = (
            gaussian_fractions
            * total_broadening
        )

        lorentzian_broadening = (
            1.0
            - gaussian_fractions
        ) * total_broadening

        clean_fids = (
            self._apply_fid_effects(
                metabolite_fids=(
                    metabolite_fids
                ),
                global_phases=(
                    global_phases
                ),
                frequency_shifts=(
                    frequency_shifts
                ),
                gaussian_broadening=(
                    gaussian_broadening
                ),
                lorentzian_broadening=(
                    lorentzian_broadening
                ),
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
            acquisition_delays_seconds=(
                acquisition_delays
            ),
            global_phases_radians=(
                global_phases
            ),
            frequency_shifts_hz=(
                frequency_shifts
            ),
            total_broadening=(
                total_broadening
            ),
            gaussian_fractions=(
                gaussian_fractions
            ),
            gaussian_broadening=(
                gaussian_broadening
            ),
            lorentzian_broadening=(
                lorentzian_broadening
            ),
        )

        self._validate_result(
            result
        )

        return result

    def _sample_concentrations(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> torch.Tensor:
        random_values = torch.randn(
            (
                batch_size,
                self.n_basis_components,
            ),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )

        concentrations = (
            self.sampling_table
            .means[None, :]
            + self.sampling_table
            .stds[None, :]
            * random_values
        )

        if (
            self.sampling_table
            .clip_negative_values
        ):
            concentrations = (
                concentrations.clamp_min(
                    self.sampling_table
                    .minimum_concentration
                )
            )

        concentrations = torch.where(
            self.sampling_table
            .enabled_mask[None, :],
            concentrations,
            torch.zeros_like(
                concentrations
            ),
        )

        return concentrations.contiguous()

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
        frequency_shifts: torch.Tensor,
        gaussian_broadening: torch.Tensor,
        lorentzian_broadening: torch.Tensor,
    ) -> torch.Tensor:
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
            * frequency_shifts[:, None]
            * time_axis
        )

        phase_factor = torch.polar(
            torch.ones_like(
                phase_angles
            ),
            phase_angles,
        )

        gaussian_decay = (
            time_axis.square()
            * gaussian_broadening[
                :,
                None,
            ].square()
        )

        lorentzian_decay = (
            time_axis.abs()
            * lorentzian_broadening[
                :,
                None,
            ]
        )

        decay_factor = torch.exp(
            -gaussian_decay
            - lorentzian_decay
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