# TARGET DEFINITION: clean water + lipids baseline
# Input: metabolites + water + lipids + receiver noise

# src/walinet/training_data/spectrum_assembly.py

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.acquisition_length import (
    simulate_acquisition_length,
)
from walinet.training_data.metabolite_noise import (
    SimulatedNoise,
)
from walinet.training_data.metabolite_simulation import (
    SimulatedMetabolites,
)
from walinet.training_data.simulation_resources import (
    SimulationPool,
)
from walinet.training_data.simulator import (
    LipidMixture,
    SampledResources,
)


@dataclass(frozen=True)
class AssembledSpectra:
    """
    Fully assembled frequency-domain simulation batch.

    Shapes
    ------
    water_spectra:
        (batch_size, n_timepoints)

        Scaled water spectra before acquisition-length simulation.

    lipid_spectra:
        (batch_size, n_timepoints)

        Scaled lipid spectra before acquisition-length simulation.

    clean_mixture_spectra:
        Clean metabolites + water + lipids before receiver noise
        and before acquisition-length simulation.

    mixture_spectra:
        Final noisy input after acquisition-length simulation.

    baseline_spectra:
        Final clean water-plus-lipid target after the same
        acquisition-length simulation as mixture_spectra.

    acquired_n_timepoints:
        (batch_size,)

        Number of actually acquired FID samples per batch item.

    projected_spectra:
        Optional result after applying the subject-specific
        frequency-domain lipid-projection operator to the final
        mixture_spectra.
    """

    metabolites: SimulatedMetabolites
    noise: SimulatedNoise

    water_spectra: torch.Tensor
    lipid_spectra: torch.Tensor

    clean_mixture_spectra: torch.Tensor

    mixture_spectra: torch.Tensor
    baseline_spectra: torch.Tensor

    acquired_n_timepoints: torch.Tensor

    projected_spectra: torch.Tensor | None

    water_scaling: torch.Tensor
    lipid_scaling: torch.Tensor

    water_subject_indices: torch.Tensor
    lipid_subject_indices: torch.Tensor

    @property
    def clean_metabolite_spectra(
        self,
    ) -> torch.Tensor:
        """
        Clean metabolite spectrum before acquisition-length
        simulation.
        """
        return self.metabolites.clean_spectra

    @property
    def target_spectra(
        self,
    ) -> torch.Tensor:
        """
        Final clean baseline target: water + lipids.
        """
        return self.baseline_spectra

    @property
    def metabolite_spectra(
        self,
    ) -> torch.Tensor:
        """
        Backward-compatible alias for the training target.

        The target is now the clean water-plus-lipid baseline,
        not a metabolite spectrum. This alias keeps the current
        SpectrumSimulator interface working until its naming is
        cleaned up separately.
        """
        return self.baseline_spectra

    @property
    def input_spectra(
        self,
    ) -> torch.Tensor:
        """
        Final network input.

        When lipid projection is enabled, use the projected result.
        Otherwise use the unprojected noisy mixture.
        """
        if self.projected_spectra is not None:
            return self.projected_spectra

        return self.mixture_spectra

    @property
    def batch_size(self) -> int:
        return int(
            self.mixture_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.mixture_spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.mixture_spectra.device


def _config_value(
    value: object,
) -> str:
    return str(
        getattr(
            value,
            "value",
            value,
        )
    )


def _validate_generator_device(
    *,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    generator_device = torch.device(
        generator.device
    )

    if generator_device.type != device.type:
        raise ValueError(
            "Generator and signals must use the same "
            "device type:\n"
            f"  generator: {generator_device}\n"
            f"  signals:   {device}"
        )

    if (
        device.type == "cuda"
        and generator_device.index is not None
        and device.index is not None
        and generator_device.index != device.index
    ):
        raise ValueError(
            "Generator and signals must use the same "
            "CUDA device:\n"
            f"  generator: {generator_device}\n"
            f"  signals:   {device}"
        )


def _sample_uniform(
    *,
    minimum: float,
    maximum: float,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    if not math.isfinite(
        minimum
    ):
        raise ValueError(
            "Uniform minimum must be finite."
        )

    if not math.isfinite(
        maximum
    ):
        raise ValueError(
            "Uniform maximum must be finite."
        )

    if maximum < minimum:
        raise ValueError(
            "Uniform maximum must be >= minimum."
        )

    if maximum == minimum:
        return torch.full(
            (batch_size,),
            fill_value=minimum,
            device=device,
            dtype=dtype,
        )

    random_values = torch.rand(
        (batch_size,),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return (
        minimum
        + (
            maximum
            - minimum
        )
        * random_values
    )


def _sample_lipid_scaling(
    *,
    config: SimulationConfig,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    minimum = float(
        config.lipids.scaling_min
    )

    maximum = float(
        config.lipids.scaling_max
    )

    distribution = _config_value(
        config
        .lipids
        .scaling_distribution
    )

    if maximum < minimum:
        raise ValueError(
            "lipids.scaling_max must be >= "
            "lipids.scaling_min."
        )

    if distribution == "uniform":
        return _sample_uniform(
            minimum=minimum,
            maximum=maximum,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
            generator=generator,
        )

    if distribution == "log_uniform":
        if minimum <= 0:
            raise ValueError(
                "Log-uniform lipid scaling requires "
                "lipids.scaling_min > 0."
            )

        if maximum <= 0:
            raise ValueError(
                "Log-uniform lipid scaling requires "
                "lipids.scaling_max > 0."
            )

        if minimum == maximum:
            return torch.full(
                (batch_size,),
                fill_value=minimum,
                device=device,
                dtype=dtype,
            )

        random_values = torch.rand(
            (batch_size,),
            generator=generator,
            device=device,
            dtype=dtype,
        )

        log_minimum = math.log(
            minimum
        )

        log_maximum = math.log(
            maximum
        )

        return torch.exp(
            log_minimum
            + (
                log_maximum
                - log_minimum
            )
            * random_values
        )

    raise ValueError(
        "Unsupported lipid scaling distribution:\n"
        f"  {distribution!r}"
    )


def _maximum_absolute_value(
    spectra: torch.Tensor,
    *,
    description: str,
) -> torch.Tensor:
    """
    Return one maximum magnitude per spectrum.

    Output shape:
        (batch_size, 1)
    """
    if spectra.ndim != 2:
        raise ValueError(
            f"{description} must have shape (B, T), "
            f"but found {tuple(spectra.shape)}."
        )

    maximum = torch.amax(
        torch.abs(
            spectra
        ),
        dim=-1,
        keepdim=True,
    )

    invalid_rows = (
        ~torch.isfinite(maximum)
        | (maximum <= 0)
    ).squeeze(-1)

    if torch.any(
        invalid_rows
    ):
        bad_indices = torch.nonzero(
            invalid_rows,
            as_tuple=False,
        ).squeeze(-1)

        raise RuntimeError(
            f"{description} contains invalid or zero-amplitude "
            "rows at batch indices:\n"
            f"  {bad_indices.detach().cpu().tolist()}"
        )

    return maximum


def _validate_complex_batch(
    *,
    tensor: torch.Tensor,
    name: str,
    expected_shape: tuple[int, int],
    expected_device: torch.device,
) -> None:
    if tuple(
        tensor.shape
    ) != expected_shape:
        raise ValueError(
            f"Unexpected shape for {name}:\n"
            f"  expected: {expected_shape}\n"
            f"  found:    {tuple(tensor.shape)}"
        )

    if tensor.device != expected_device:
        raise ValueError(
            f"{name} is on {tensor.device}, "
            f"but expected {expected_device}."
        )

    if not torch.is_complex(
        tensor
    ):
        raise TypeError(
            f"{name} must be complex-valued."
        )


def assemble_spectra(
    *,
    sampled: SampledResources,
    lipid_mixture: LipidMixture,
    metabolites: SimulatedMetabolites,
    noise: SimulatedNoise,
    pool: SimulationPool,
    config: SimulationConfig,
    generator: torch.Generator,
) -> AssembledSpectra:
    """
    Scale and combine frequency-domain simulation components.

    Processing
    ----------
    1. Use the clean metabolite spectrum as amplitude reference.
    2. Normalize each sampled water spectrum by its own maximum.
    3. Normalize each mixed lipid spectrum by its own maximum.
    4. Apply random water and lipid scaling factors.
    5. Add clean metabolites, water, and lipids.
    6. Add receiver noise to the complete input mixture.
    7. Build the clean water-plus-lipid baseline target.
    8. Apply the same sampled acquisition length and zero-filling
       operation to input and target.
    9. Optionally apply the frequency-domain lipid projection to
       the final input.

    Water and lipids are already fft-shifted spectra when entering
    this function.
    """
    device = sampled.device

    _validate_generator_device(
        generator=generator,
        device=device,
    )

    batch_size = sampled.batch_size
    n_timepoints = sampled.n_timepoints

    expected_shape = (
        batch_size,
        n_timepoints,
    )

    _validate_complex_batch(
        tensor=sampled.water_spectra,
        name="sampled.water_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=lipid_mixture.mixed_spectra,
        name="lipid_mixture.mixed_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=metabolites.clean_spectra,
        name="metabolites.clean_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    _validate_complex_batch(
        tensor=noise.noise_spectra,
        name="noise.noise_spectra",
        expected_shape=expected_shape,
        expected_device=device,
    )

    if metabolites.batch_size != batch_size:
        raise ValueError(
            "Metabolite batch size does not match "
            "the sampled-resource batch size."
        )

    if metabolites.n_timepoints != n_timepoints:
        raise ValueError(
            "Metabolite spectrum length does not match "
            "the sampled resources."
        )

    if noise.batch_size != batch_size:
        raise ValueError(
            "Noise batch size does not match "
            "the sampled resources."
        )

    if noise.n_timepoints != n_timepoints:
        raise ValueError(
            "Noise spectrum length does not match "
            "the sampled resources."
        )

    if pool.device != device:
        raise ValueError(
            "SimulationPool and sampled resources must use "
            "the same device."
        )

    if pool.n_timepoints != n_timepoints:
        raise ValueError(
            "SimulationPool and sampled resources have "
            "different spectrum lengths."
        )

    raw_water_spectra = (
        sampled.water_spectra
    )

    raw_lipid_spectra = (
        lipid_mixture.mixed_spectra
    )

    clean_metabolite_spectra = (
        metabolites.clean_spectra
    )

    metabolite_maximum = (
        _maximum_absolute_value(
            clean_metabolite_spectra,
            description=(
                "Clean metabolite spectra"
            ),
        )
    )

    water_maximum = (
        _maximum_absolute_value(
            raw_water_spectra,
            description="Water spectra",
        )
    )

    lipid_maximum = (
        _maximum_absolute_value(
            raw_lipid_spectra,
            description="Lipid spectra",
        )
    )

    real_dtype = (
        clean_metabolite_spectra
        .real
        .dtype
    )

    water_scaling = _sample_uniform(
        minimum=float(
            config.water.scaling_min
        ),
        maximum=float(
            config.water.scaling_max
        ),
        batch_size=batch_size,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )

    lipid_scaling = (
        _sample_lipid_scaling(
            config=config,
            batch_size=batch_size,
            device=device,
            dtype=real_dtype,
            generator=generator,
        )
    )

    water_spectra = (
        raw_water_spectra
        / water_maximum
        * metabolite_maximum
        * water_scaling[:, None]
    ).contiguous()

    lipid_spectra = (
        raw_lipid_spectra
        / lipid_maximum
        * metabolite_maximum
        * lipid_scaling[:, None]
    ).contiguous()

    clean_baseline_spectra = (
        water_spectra
        + lipid_spectra
    ).contiguous()

    clean_mixture_spectra = (
        clean_metabolite_spectra
        + clean_baseline_spectra
    ).contiguous()

    pre_acquisition_input_spectra = (
        clean_mixture_spectra
        + noise.noise_spectra
    ).contiguous()

    pre_acquisition_target_spectra = (
        clean_baseline_spectra
    )

    # Input and target are transformed together so that every
    # batch item receives exactly the same sampled acquisition
    # length in both tensors.
    #
    # Input:
    #     metabolites + water + lipids + receiver noise
    #
    # Target:
    #     water + lipids
    stacked_spectra = torch.stack(
        (
            pre_acquisition_input_spectra,
            pre_acquisition_target_spectra,
        ),
        dim=1,
    )

    acquisition_result = (
        simulate_acquisition_length(
            spectra=stacked_spectra,
            config=config,
            generator=generator,
        )
    )

    mixture_spectra = (
        acquisition_result
        .spectra[:, 0, :]
        .contiguous()
    )

    baseline_spectra = (
        acquisition_result
        .spectra[:, 1, :]
        .contiguous()
    )

    acquired_n_timepoints = (
        acquisition_result
        .acquired_n_timepoints
        .contiguous()
    )

    projected_spectra: (
        torch.Tensor | None
    )

    if config.lipid_projection.enabled:
        operators = (
            pool
            .lipid_projection_operators
        )

        if operators is None:
            raise RuntimeError(
                "Lipid projection is enabled, but the "
                "SimulationPool contains no projection operators."
            )

        if operators.device != device:
            raise ValueError(
                "Projection operators and simulated spectra "
                "must use the same device."
            )

        expected_operator_pool_shape = (
            pool.n_subjects,
            n_timepoints,
            n_timepoints,
        )

        if (
            tuple(operators.shape)
            != expected_operator_pool_shape
        ):
            raise RuntimeError(
                "Unexpected projection-operator pool shape:\n"
                f"  expected: {expected_operator_pool_shape}\n"
                f"  found:    {tuple(operators.shape)}"
            )

        # Apply each subject-specific projection operator only once
        # to all spectra belonging to that subject.
        #
        # This avoids materializing a tensor with shape
        #
        #     (batch_size, n_timepoints, n_timepoints)
        #
        # which would require approximately 20 GB for
        # batch_size=3500, n_timepoints=840, complex64.
        projected_spectra = torch.empty_like(
            mixture_spectra
        )

        unique_subject_indices = torch.unique(
            sampled.lipid_subject_indices
        )

        for subject_index_tensor in unique_subject_indices:
            subject_index = int(
                subject_index_tensor.item()
            )

            batch_indices = torch.nonzero(
                sampled.lipid_subject_indices
                == subject_index_tensor,
                as_tuple=False,
            ).squeeze(-1)

            if batch_indices.numel() == 0:
                continue

            subject_spectra = (
                mixture_spectra.index_select(
                    0,
                    batch_indices,
                )
            )

            subject_operator = (
                operators[subject_index]
            )

            expected_operator_shape = (
                n_timepoints,
                n_timepoints,
            )

            if (
                tuple(subject_operator.shape)
                != expected_operator_shape
            ):
                raise RuntimeError(
                    "Unexpected subject projection-operator shape:\n"
                    f"  subject index: {subject_index}\n"
                    f"  expected:      {expected_operator_shape}\n"
                    f"  found:         "
                    f"{tuple(subject_operator.shape)}"
                )

            # Row-vector convention:
            #
            #     projected = spectrum @ operator
            projected_subject_spectra = (
                subject_spectra
                @ subject_operator
            )

            projected_spectra.index_copy_(
                0,
                batch_indices,
                projected_subject_spectra,
            )

        projected_spectra = (
            projected_spectra.contiguous()
        )

    else:
        projected_spectra = None

    output_tensors = {
        "water_spectra":
            water_spectra,

        "lipid_spectra":
            lipid_spectra,

        "clean_mixture_spectra":
            clean_mixture_spectra,

        "mixture_spectra":
            mixture_spectra,

        "baseline_spectra":
            baseline_spectra,
    }

    if projected_spectra is not None:
        output_tensors[
            "projected_spectra"
        ] = projected_spectra

    for name, tensor in output_tensors.items():
        if not torch.isfinite(
            tensor.real
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite real values."
            )

        if not torch.isfinite(
            tensor.imag
        ).all():
            raise RuntimeError(
                f"{name} contains non-finite imaginary values."
            )

    if tuple(
        acquired_n_timepoints.shape
    ) != (batch_size,):
        raise RuntimeError(
            "Unexpected acquired_n_timepoints shape:\n"
            f"  expected: {(batch_size,)}\n"
            f"  found:    "
            f"{tuple(acquired_n_timepoints.shape)}"
        )

    if acquired_n_timepoints.device != device:
        raise RuntimeError(
            "acquired_n_timepoints is on "
            f"{acquired_n_timepoints.device}, "
            f"but expected {device}."
        )

    if torch.any(
        acquired_n_timepoints
        < config.acquisition.min_acquired_n_timepoints
    ):
        raise RuntimeError(
            "At least one sampled acquisition length is below "
            "the configured minimum."
        )

    if torch.any(
        acquired_n_timepoints
        > config.acquisition.max_acquired_n_timepoints
    ):
        raise RuntimeError(
            "At least one sampled acquisition length is above "
            "the configured maximum."
        )

    return AssembledSpectra(
        metabolites=metabolites,
        noise=noise,
        water_spectra=water_spectra,
        lipid_spectra=lipid_spectra,
        clean_mixture_spectra=(
            clean_mixture_spectra
        ),
        mixture_spectra=(
            mixture_spectra
        ),
        baseline_spectra=(
            baseline_spectra
        ),
        acquired_n_timepoints=(
            acquired_n_timepoints
        ),
        projected_spectra=(
            projected_spectra
        ),
        water_scaling=(
            water_scaling.contiguous()
        ),
        lipid_scaling=(
            lipid_scaling.contiguous()
        ),
        water_subject_indices=(
            sampled
            .water_subject_indices
            .contiguous()
        ),
        lipid_subject_indices=(
            sampled
            .lipid_subject_indices
            .contiguous()
        ),
    )