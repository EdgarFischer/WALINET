# src/walinet/training_data/simulator.py

from __future__ import annotations

from dataclasses import dataclass

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.simulation_resources import (
    SimulationPool,
)


@dataclass(frozen=True)
class SampledResources:
    """
    Frequency-domain water and lipid resources sampled for one batch.

    Shapes
    ------
    water_spectra:
        (batch_size, n_timepoints)

    lipid_spectra:
        (
            batch_size,
            n_random_lipid_spectra,
            n_timepoints,
        )

    water_subject_indices:
        (batch_size,)

    lipid_subject_indices:
        (batch_size,)

    water_resource_indices:
        (batch_size,)

    lipid_resource_indices:
        (batch_size, n_random_lipid_spectra)
    """

    water_spectra: torch.Tensor
    lipid_spectra: torch.Tensor

    water_subject_indices: torch.Tensor
    lipid_subject_indices: torch.Tensor

    water_resource_indices: torch.Tensor
    lipid_resource_indices: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(
            self.water_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.water_spectra.shape[-1]
        )

    @property
    def n_random_lipid_spectra(self) -> int:
        return int(
            self.lipid_spectra.shape[1]
        )

    @property
    def device(self) -> torch.device:
        return self.water_spectra.device


@dataclass(frozen=True)
class LipidMixture:
    """
    Weighted mixture of sampled lipid spectra.

    Shapes
    ------
    mixed_spectra:
        (batch_size, n_timepoints)

    weights:
        (batch_size, n_random_lipid_spectra)
    """

    mixed_spectra: torch.Tensor
    weights: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(
            self.mixed_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.mixed_spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.mixed_spectra.device


def _config_value(
    value: object,
) -> str:
    """
    Convert a normal string or Enum-like config value to a string.
    """
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
            "Generator and resource pool must use the same "
            "device type:\n"
            f"  generator: {generator_device}\n"
            f"  pool:      {device}"
        )

    if (
        device.type == "cuda"
        and generator_device.index is not None
        and device.index is not None
        and generator_device.index != device.index
    ):
        raise ValueError(
            "Generator and resource pool must use the same "
            "CUDA device:\n"
            f"  generator: {generator_device}\n"
            f"  pool:      {device}"
        )


def _validate_pool(
    pool: SimulationPool,
) -> None:
    if pool.water_spectra.ndim != 2:
        raise ValueError(
            "pool.water_spectra must have shape (N, T), "
            f"but found {tuple(pool.water_spectra.shape)}."
        )

    if pool.lipid_spectra.ndim != 2:
        raise ValueError(
            "pool.lipid_spectra must have shape (N, T), "
            f"but found {tuple(pool.lipid_spectra.shape)}."
        )

    if not torch.is_complex(
        pool.water_spectra
    ):
        raise TypeError(
            "pool.water_spectra must be complex-valued."
        )

    if not torch.is_complex(
        pool.lipid_spectra
    ):
        raise TypeError(
            "pool.lipid_spectra must be complex-valued."
        )

    if (
        pool.water_spectra.shape[-1]
        != pool.n_timepoints
    ):
        raise ValueError(
            "Water spectrum length does not match "
            "pool.n_timepoints."
        )

    if (
        pool.lipid_spectra.shape[-1]
        != pool.n_timepoints
    ):
        raise ValueError(
            "Lipid spectrum length does not match "
            "pool.n_timepoints."
        )

    expected_offset_shape = (
        pool.n_subjects + 1,
    )

    if (
        tuple(pool.water_offsets.shape)
        != expected_offset_shape
    ):
        raise ValueError(
            "Unexpected water-offset shape:\n"
            f"  expected: {expected_offset_shape}\n"
            f"  found:    {tuple(pool.water_offsets.shape)}"
        )

    if (
        tuple(pool.lipid_offsets.shape)
        != expected_offset_shape
    ):
        raise ValueError(
            "Unexpected lipid-offset shape:\n"
            f"  expected: {expected_offset_shape}\n"
            f"  found:    {tuple(pool.lipid_offsets.shape)}"
        )

    if pool.n_subjects < 1:
        raise ValueError(
            "SimulationPool must contain at least one subject."
        )

    if int(
        pool.water_offsets[0].item()
    ) != 0:
        raise ValueError(
            "water_offsets must begin with zero."
        )

    if int(
        pool.lipid_offsets[0].item()
    ) != 0:
        raise ValueError(
            "lipid_offsets must begin with zero."
        )

    if int(
        pool.water_offsets[-1].item()
    ) != int(
        pool.water_spectra.shape[0]
    ):
        raise ValueError(
            "Last water offset does not match the number "
            "of water spectra."
        )

    if int(
        pool.lipid_offsets[-1].item()
    ) != int(
        pool.lipid_spectra.shape[0]
    ):
        raise ValueError(
            "Last lipid offset does not match the number "
            "of lipid spectra."
        )

    if torch.any(
        pool.water_counts <= 0
    ):
        raise ValueError(
            "Every subject must contain at least one "
            "water spectrum."
        )

    if torch.any(
        pool.lipid_counts <= 0
    ):
        raise ValueError(
            "Every subject must contain at least one "
            "lipid spectrum."
        )

    pool_tensors = {
        "water_spectra":
            pool.water_spectra,

        "lipid_spectra":
            pool.lipid_spectra,

        "water_offsets":
            pool.water_offsets,

        "lipid_offsets":
            pool.lipid_offsets,

        "native_lengths":
            pool.native_lengths,
    }

    for name, tensor in pool_tensors.items():
        if tensor.device != pool.device:
            raise ValueError(
                f"pool.{name} is on {tensor.device}, "
                f"but expected {pool.device}."
            )


def _sample_subject_indices(
    *,
    n_subjects: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.randint(
        low=0,
        high=n_subjects,
        size=(batch_size,),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )


def _sample_one_resource_per_subject(
    *,
    offsets: torch.Tensor,
    subject_indices: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample one resource uniformly from each selected subject.

    Returns global indices into the flat resource pool.
    """
    starts = offsets[:-1].index_select(
        0,
        subject_indices,
    )

    counts = (
        offsets[1:]
        - offsets[:-1]
    ).index_select(
        0,
        subject_indices,
    )

    random_values = torch.rand(
        subject_indices.shape,
        generator=generator,
        device=subject_indices.device,
        dtype=torch.float32,
    )

    local_indices = torch.floor(
        random_values
        * counts.to(
            dtype=torch.float32
        )
    ).to(
        dtype=torch.int64
    )

    return (
        starts
        + local_indices
    ).contiguous()


def _sample_multiple_resources_per_subject(
    *,
    offsets: torch.Tensor,
    subject_indices: torch.Tensor,
    n_resources: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample several resources with replacement from each subject.

    Returns global indices with shape:

        (batch_size, n_resources)
    """
    starts = offsets[:-1].index_select(
        0,
        subject_indices,
    )

    counts = (
        offsets[1:]
        - offsets[:-1]
    ).index_select(
        0,
        subject_indices,
    )

    random_values = torch.rand(
        (
            subject_indices.shape[0],
            n_resources,
        ),
        generator=generator,
        device=subject_indices.device,
        dtype=torch.float32,
    )

    local_indices = torch.floor(
        random_values
        * counts[:, None].to(
            dtype=torch.float32
        )
    ).to(
        dtype=torch.int64
    )

    return (
        starts[:, None]
        + local_indices
    ).contiguous()


class SimulationResourceSampler:
    """
    Sample water and lipid spectra from a SimulationPool.

    Subjects are sampled uniformly. Resources are then sampled
    uniformly within the selected subject.

    Lipid spectra are sampled with replacement, matching the
    previous simulator's random resource selection.
    """

    def __init__(
        self,
        *,
        pool: SimulationPool,
        config: SimulationConfig,
    ) -> None:
        _validate_pool(
            pool
        )

        self.pool = pool

        self.mixing = _config_value(
            config
            .subject_sampling
            .mixing
        )

        self.n_random_lipid_spectra = int(
            config
            .lipids
            .n_random_fids
        )

        if self.n_random_lipid_spectra <= 0:
            raise ValueError(
                "lipids.n_random_fids must be > 0."
            )

        supported_mixing = {
            "same_subject",
            "separate_water_lipid_subjects",
        }

        if self.mixing not in supported_mixing:
            raise ValueError(
                "Unsupported subject-sampling mode:\n"
                f"  found: {self.mixing!r}\n"
                f"  supported: {sorted(supported_mixing)}"
            )

    def sample(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
    ) -> SampledResources:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be > 0."
            )

        pool = self.pool
        device = pool.device

        _validate_generator_device(
            generator=generator,
            device=device,
        )

        water_subject_indices = (
            _sample_subject_indices(
                n_subjects=pool.n_subjects,
                batch_size=batch_size,
                device=device,
                generator=generator,
            )
        )

        if self.mixing == "same_subject":
            lipid_subject_indices = (
                water_subject_indices.clone()
            )

        else:
            lipid_subject_indices = (
                _sample_subject_indices(
                    n_subjects=pool.n_subjects,
                    batch_size=batch_size,
                    device=device,
                    generator=generator,
                )
            )

        water_resource_indices = (
            _sample_one_resource_per_subject(
                offsets=pool.water_offsets,
                subject_indices=(
                    water_subject_indices
                ),
                generator=generator,
            )
        )

        lipid_resource_indices = (
            _sample_multiple_resources_per_subject(
                offsets=pool.lipid_offsets,
                subject_indices=(
                    lipid_subject_indices
                ),
                n_resources=(
                    self.n_random_lipid_spectra
                ),
                generator=generator,
            )
        )

        water_spectra = pool.water_spectra[
            water_resource_indices
        ]

        lipid_spectra = pool.lipid_spectra[
            lipid_resource_indices
        ]

        expected_water_shape = (
            batch_size,
            pool.n_timepoints,
        )

        expected_lipid_shape = (
            batch_size,
            self.n_random_lipid_spectra,
            pool.n_timepoints,
        )

        if (
            tuple(water_spectra.shape)
            != expected_water_shape
        ):
            raise RuntimeError(
                "Unexpected sampled water shape:\n"
                f"  expected: {expected_water_shape}\n"
                f"  found:    {tuple(water_spectra.shape)}"
            )

        if (
            tuple(lipid_spectra.shape)
            != expected_lipid_shape
        ):
            raise RuntimeError(
                "Unexpected sampled lipid shape:\n"
                f"  expected: {expected_lipid_shape}\n"
                f"  found:    {tuple(lipid_spectra.shape)}"
            )

        return SampledResources(
            water_spectra=(
                water_spectra.contiguous()
            ),
            lipid_spectra=(
                lipid_spectra.contiguous()
            ),
            water_subject_indices=(
                water_subject_indices.contiguous()
            ),
            lipid_subject_indices=(
                lipid_subject_indices.contiguous()
            ),
            water_resource_indices=(
                water_resource_indices.contiguous()
            ),
            lipid_resource_indices=(
                lipid_resource_indices.contiguous()
            ),
        )


def mix_sampled_lipid_spectra(
    *,
    sampled: SampledResources,
    generator: torch.Generator,
) -> LipidMixture:
    """
    Mix sampled lipid spectra using positive normalized weights.

    Because the Fourier transform is linear,

        FFT(sum_i w_i * fid_i)
        =
        sum_i w_i * FFT(fid_i),

    this is mathematically identical to mixing the corresponding
    FIDs first and transforming the result afterwards.
    """
    lipid_spectra = sampled.lipid_spectra

    if lipid_spectra.ndim != 3:
        raise ValueError(
            "sampled.lipid_spectra must have shape "
            "(B, N, T), but found "
            f"{tuple(lipid_spectra.shape)}."
        )

    if not torch.is_complex(
        lipid_spectra
    ):
        raise TypeError(
            "sampled.lipid_spectra must be complex-valued."
        )

    _validate_generator_device(
        generator=generator,
        device=lipid_spectra.device,
    )

    batch_size = int(
        lipid_spectra.shape[0]
    )

    n_lipid_spectra = int(
        lipid_spectra.shape[1]
    )

    if n_lipid_spectra <= 0:
        raise ValueError(
            "At least one lipid spectrum is required."
        )

    weights = torch.rand(
        (
            batch_size,
            n_lipid_spectra,
        ),
        generator=generator,
        device=lipid_spectra.device,
        dtype=lipid_spectra.real.dtype,
    )

    weight_sums = weights.sum(
        dim=-1,
        keepdim=True,
    )

    if torch.any(
        weight_sums <= 0
    ):
        raise RuntimeError(
            "Generated lipid weights have an invalid sum."
        )

    weights = (
        weights
        / weight_sums
    )

    mixed_spectra = torch.sum(
        lipid_spectra
        * weights[:, :, None],
        dim=1,
    )

    if not torch.isfinite(
        mixed_spectra.real
    ).all():
        raise RuntimeError(
            "Mixed lipid spectra contain non-finite "
            "real values."
        )

    if not torch.isfinite(
        mixed_spectra.imag
    ).all():
        raise RuntimeError(
            "Mixed lipid spectra contain non-finite "
            "imaginary values."
        )

    return LipidMixture(
        mixed_spectra=(
            mixed_spectra.contiguous()
        ),
        weights=weights.contiguous(),
    )