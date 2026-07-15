# FINAL TRAINER-READY VERSION
# Includes max_retries, normalization, network_input, network_target, and simulate_raw.

# src/walinet/training_data/spectrum_simulator.py

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.metabolite_noise import (
    SimulatedNoise,
    simulate_receiver_noise,
)
from walinet.training_data.metabolite_simulation import (
    MetaboliteSimulator,
    SimulatedMetabolites,
)
from walinet.training_data.simulation_resources import (
    SimulationPool,
)
from walinet.training_data.simulator import (
    LipidMixture,
    SampledResources,
    SimulationResourceSampler,
    mix_sampled_lipid_spectra,
)
from walinet.training_data.spectrum_assembly import (
    AssembledSpectra,
    assemble_spectra,
)


class RetryableSimulationError(RuntimeError):
    """
    Numerical simulation error for which the complete batch should
    be discarded and newly sampled.
    """


_RETRYABLE_ERROR_MARKERS = (
    "non-finite",
    "nan",
    "infinite",
    "zero-amplitude",
    "zero amplitude",
    "zero standard deviation",
)


def _is_retryable_simulation_error(
    error: BaseException,
) -> bool:
    if isinstance(
        error,
        RetryableSimulationError,
    ):
        return True

    message = str(
        error
    ).lower()

    return any(
        marker in message
        for marker in _RETRYABLE_ERROR_MARKERS
    )


def complex_spectra_to_channels(
    spectra: torch.Tensor,
) -> torch.Tensor:
    """
    Convert complex spectra from (B, T) to real-valued network
    tensors with shape (B, 2, T).

    Channel 0:
        Real part.

    Channel 1:
        Imaginary part.
    """
    if spectra.ndim != 2:
        raise ValueError(
            "spectra must have shape (B, T), "
            f"but found {tuple(spectra.shape)}."
        )

    if not torch.is_complex(
        spectra
    ):
        raise TypeError(
            "spectra must be complex-valued."
        )

    return torch.stack(
        (
            spectra.real,
            spectra.imag,
        ),
        dim=1,
    ).contiguous()


@dataclass(frozen=True)
class SimulatedSpectrumBatch:
    """
    Complete unnormalized output of one vectorized simulation call.
    """

    sampled_resources: SampledResources
    lipid_mixture: LipidMixture

    metabolites: SimulatedMetabolites
    noise: SimulatedNoise

    assembled: AssembledSpectra

    @property
    def input_spectra(
        self,
    ) -> torch.Tensor:
        return self.assembled.input_spectra

    @property
    def target_spectra(
        self,
    ) -> torch.Tensor:
        return self.assembled.metabolite_spectra

    @property
    def acquired_n_timepoints(
        self,
    ) -> torch.Tensor:
        return (
            self.assembled
            .acquired_n_timepoints
        )

    @property
    def snr(
        self,
    ) -> torch.Tensor:
        return self.noise.snr

    @property
    def concentrations(
        self,
    ) -> torch.Tensor:
        return self.metabolites.concentrations

    @property
    def water_scaling(
        self,
    ) -> torch.Tensor:
        return self.assembled.water_scaling

    @property
    def lipid_scaling(
        self,
    ) -> torch.Tensor:
        return self.assembled.lipid_scaling

    @property
    def batch_size(self) -> int:
        return int(
            self.input_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.input_spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.input_spectra.device


@dataclass(frozen=True)
class PreparedSpectrumBatch:
    """
    Final trainer-ready batch.

    Normalization
    -------------
    For every batch item, the normalization factor is calculated
    only from the final input spectrum:

        scale = max(abs(input))

    The identical scale is used for input and target. This preserves
    their relative amplitudes.

    Shapes
    ------
    normalized_input_spectra:
        (B, T), complex

    normalized_target_spectra:
        (B, T), complex

    network_input:
        (B, 2, T), float

    network_target:
        (B, 2, T), float

    normalization_scale:
        (B, 1), float
    """

    raw: SimulatedSpectrumBatch

    normalized_input_spectra: torch.Tensor
    normalized_target_spectra: torch.Tensor

    network_input: torch.Tensor
    network_target: torch.Tensor

    normalization_scale: torch.Tensor

    retries_used: int

    @property
    def raw_input_spectra(
        self,
    ) -> torch.Tensor:
        return self.raw.input_spectra

    @property
    def raw_target_spectra(
        self,
    ) -> torch.Tensor:
        return self.raw.target_spectra

    @property
    def acquired_n_timepoints(
        self,
    ) -> torch.Tensor:
        return self.raw.acquired_n_timepoints

    @property
    def snr(
        self,
    ) -> torch.Tensor:
        return self.raw.snr

    @property
    def concentrations(
        self,
    ) -> torch.Tensor:
        return self.raw.concentrations

    @property
    def water_scaling(
        self,
    ) -> torch.Tensor:
        return self.raw.water_scaling

    @property
    def lipid_scaling(
        self,
    ) -> torch.Tensor:
        return self.raw.lipid_scaling

    @property
    def batch_size(self) -> int:
        return int(
            self.network_input.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.network_input.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.network_input.device


class SpectrumSimulator:
    """
    Complete vectorized WALINET on-the-fly spectrum simulator.

    simulate_raw():
        Return the complete unnormalized complex simulation.

    simulate():
        Return the trainer-ready, normalized real/imaginary tensors.
        On rare non-finite numerical failures, the complete batch is
        discarded and newly sampled.
    """

    def __init__(
        self,
        *,
        pool: SimulationPool,
        metabolite_simulator: MetaboliteSimulator,
        config: SimulationConfig,
        max_retries: int = 3,
    ) -> None:
        if max_retries < 0:
            raise ValueError(
                "max_retries must be >= 0."
            )

        self.pool = pool
        self.metabolite_simulator = (
            metabolite_simulator
        )
        self.config = config
        self.max_retries = int(
            max_retries
        )

        self.discarded_batches = 0
        self.discarded_spectra = 0

        self._validate_components()

        self.resource_sampler = (
            SimulationResourceSampler(
                pool=self.pool,
                config=self.config,
            )
        )

    @property
    def device(self) -> torch.device:
        return self.pool.device

    @property
    def n_timepoints(self) -> int:
        return self.pool.n_timepoints

    @torch.no_grad()
    def simulate_raw(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> SimulatedSpectrumBatch:
        """
        Simulate one complete unnormalized complex batch.
        """
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        self._validate_generator_device(
            generator
        )

        sampled_resources = (
            self.resource_sampler.sample(
                batch_size=batch_size,
                generator=generator,
            )
        )

        lipid_mixture = (
            mix_sampled_lipid_spectra(
                sampled=sampled_resources,
                generator=generator,
            )
        )

        metabolites = (
            self.metabolite_simulator.simulate(
                batch_size=batch_size,
                generator=generator,
            )
        )

        noise = simulate_receiver_noise(
            metabolites=metabolites,
            config=self.config,
            generator=generator,
        )

        assembled = assemble_spectra(
            sampled=sampled_resources,
            lipid_mixture=lipid_mixture,
            metabolites=metabolites,
            noise=noise,
            pool=self.pool,
            config=self.config,
            generator=generator,
        )

        result = SimulatedSpectrumBatch(
            sampled_resources=(
                sampled_resources
            ),
            lipid_mixture=(
                lipid_mixture
            ),
            metabolites=metabolites,
            noise=noise,
            assembled=assembled,
        )

        self._validate_raw_result(
            result=result,
            expected_batch_size=batch_size,
        )

        return result

    @torch.no_grad()
    def simulate(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
        max_retries: int | None = None,
    ) -> PreparedSpectrumBatch:
        """
        Generate one final trainer-ready batch.

        Retryable numerical failures discard the complete batch.
        Other errors are raised immediately.
        """
        retries = (
            self.max_retries
            if max_retries is None
            else int(max_retries)
        )

        if retries < 0:
            raise ValueError(
                "max_retries must be >= 0."
            )

        for attempt in range(
            retries + 1
        ):
            try:
                raw = self.simulate_raw(
                    batch_size=batch_size,
                    generator=generator,
                )

                return self._prepare_batch(
                    raw=raw,
                    retries_used=attempt,
                )

            except (
                RuntimeError,
                FloatingPointError,
            ) as error:
                if not _is_retryable_simulation_error(
                    error
                ):
                    raise

                self.discarded_batches += 1
                self.discarded_spectra += (
                    batch_size
                )

                if attempt >= retries:
                    raise RuntimeError(
                        "Simulation repeatedly produced an "
                        "invalid numerical batch after "
                        f"{retries + 1} attempts."
                    ) from error

        raise AssertionError(
            "Unreachable code."
        )

    def _prepare_batch(
        self,
        *,
        raw: SimulatedSpectrumBatch,
        retries_used: int,
    ) -> PreparedSpectrumBatch:
        """
        Apply input-based max-absolute normalization, convert to
        real/imaginary channels, and run the final finite check.
        """
        raw_input = raw.input_spectra
        raw_target = raw.target_spectra

        normalization_scale = torch.amax(
            torch.abs(
                raw_input
            ),
            dim=-1,
            keepdim=True,
        )

        invalid_scale = (
            ~torch.isfinite(
                normalization_scale
            )
            | (
                normalization_scale
                <= 0
            )
        )

        if bool(
            torch.any(
                invalid_scale
            )
        ):
            raise RetryableSimulationError(
                "Normalization scale is zero or non-finite."
            )

        normalized_input = (
            raw_input
            / normalization_scale
        ).contiguous()

        normalized_target = (
            raw_target
            / normalization_scale
        ).contiguous()

        network_input = (
            complex_spectra_to_channels(
                normalized_input
            )
        )

        network_target = (
            complex_spectra_to_channels(
                normalized_target
            )
        )

        final_is_finite = (
            torch.isfinite(
                network_input
            ).all()
            & torch.isfinite(
                network_target
            ).all()
            & torch.isfinite(
                normalization_scale
            ).all()
        )

        if not bool(
            final_is_finite
        ):
            raise RetryableSimulationError(
                "Final normalized input, target, or "
                "normalization scale contains non-finite values."
            )

        result = PreparedSpectrumBatch(
            raw=raw,
            normalized_input_spectra=(
                normalized_input
            ),
            normalized_target_spectra=(
                normalized_target
            ),
            network_input=network_input,
            network_target=network_target,
            normalization_scale=(
                normalization_scale.contiguous()
            ),
            retries_used=int(
                retries_used
            ),
        )

        self._validate_prepared_result(
            result=result
        )

        return result

    def _validate_components(
        self,
    ) -> None:
        if (
            self.metabolite_simulator.device
            != self.pool.device
        ):
            raise ValueError(
                "Metabolite simulator and simulation pool "
                "must use the same device:\n"
                f"  metabolite simulator: "
                f"{self.metabolite_simulator.device}\n"
                f"  simulation pool:      "
                f"{self.pool.device}"
            )

        if (
            self.metabolite_simulator.n_timepoints
            != self.pool.n_timepoints
        ):
            raise ValueError(
                "Metabolite simulator and simulation pool "
                "must use the same number of timepoints:\n"
                f"  metabolite simulator: "
                f"{self.metabolite_simulator.n_timepoints}\n"
                f"  simulation pool:      "
                f"{self.pool.n_timepoints}"
            )

        expected_n_timepoints = int(
            self.config
            .acquisition
            .n_timepoints
        )

        if (
            self.pool.n_timepoints
            != expected_n_timepoints
        ):
            raise ValueError(
                "Simulation pool and configuration use "
                "different numbers of timepoints:\n"
                f"  simulation pool: {self.pool.n_timepoints}\n"
                f"  configuration:  {expected_n_timepoints}"
            )

        expected_bandwidth_hz = float(
            self.config
            .acquisition
            .bandwidth_hz
        )

        if not math.isclose(
            float(self.pool.bandwidth_hz),
            expected_bandwidth_hz,
            rel_tol=1e-6,
            abs_tol=1e-6,
        ):
            raise ValueError(
                "Simulation pool and configuration use "
                "different bandwidths:\n"
                f"  simulation pool: "
                f"{self.pool.bandwidth_hz} Hz\n"
                f"  configuration:  "
                f"{expected_bandwidth_hz} Hz"
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
                "the same CUDA device:\n"
                f"  generator: {generator_device}\n"
                f"  simulator: {self.device}"
            )

    def _validate_raw_result(
        self,
        *,
        result: SimulatedSpectrumBatch,
        expected_batch_size: int,
    ) -> None:
        expected_spectrum_shape = (
            expected_batch_size,
            self.n_timepoints,
        )

        if tuple(
            result.input_spectra.shape
        ) != expected_spectrum_shape:
            raise RuntimeError(
                "Unexpected final input shape:\n"
                f"  expected: {expected_spectrum_shape}\n"
                f"  found:    "
                f"{tuple(result.input_spectra.shape)}"
            )

        if tuple(
            result.target_spectra.shape
        ) != expected_spectrum_shape:
            raise RuntimeError(
                "Unexpected final target shape:\n"
                f"  expected: {expected_spectrum_shape}\n"
                f"  found:    "
                f"{tuple(result.target_spectra.shape)}"
            )

        if tuple(
            result.acquired_n_timepoints.shape
        ) != (expected_batch_size,):
            raise RuntimeError(
                "Unexpected acquisition-length shape."
            )

        if tuple(
            result.snr.shape
        ) != (expected_batch_size,):
            raise RuntimeError(
                "Unexpected SNR shape."
            )

        tensors_with_device = {
            "input_spectra":
                result.input_spectra,

            "target_spectra":
                result.target_spectra,

            "acquired_n_timepoints":
                result.acquired_n_timepoints,

            "snr":
                result.snr,

            "concentrations":
                result.concentrations,
        }

        for name, tensor in tensors_with_device.items():
            if tensor.device != self.device:
                raise RuntimeError(
                    f"{name} is on {tensor.device}, "
                    f"but expected {self.device}."
                )

    def _validate_prepared_result(
        self,
        *,
        result: PreparedSpectrumBatch,
    ) -> None:
        expected_complex_shape = (
            result.batch_size,
            self.n_timepoints,
        )

        expected_channel_shape = (
            result.batch_size,
            2,
            self.n_timepoints,
        )

        expected_scale_shape = (
            result.batch_size,
            1,
        )

        if tuple(
            result.normalized_input_spectra.shape
        ) != expected_complex_shape:
            raise RuntimeError(
                "Unexpected normalized input shape."
            )

        if tuple(
            result.normalized_target_spectra.shape
        ) != expected_complex_shape:
            raise RuntimeError(
                "Unexpected normalized target shape."
            )

        if tuple(
            result.network_input.shape
        ) != expected_channel_shape:
            raise RuntimeError(
                "Unexpected network input shape."
            )

        if tuple(
            result.network_target.shape
        ) != expected_channel_shape:
            raise RuntimeError(
                "Unexpected network target shape."
            )

        if tuple(
            result.normalization_scale.shape
        ) != expected_scale_shape:
            raise RuntimeError(
                "Unexpected normalization-scale shape."
            )

        for tensor in (
            result.normalized_input_spectra,
            result.normalized_target_spectra,
            result.network_input,
            result.network_target,
            result.normalization_scale,
        ):
            if tensor.device != self.device:
                raise RuntimeError(
                    "Prepared output is on the wrong device."
                )