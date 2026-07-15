# src/walinet/training_data/acquisition_length.py

from __future__ import annotations

from dataclasses import dataclass

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)


@dataclass(frozen=True)
class AcquisitionLengthResult:
    """
    Result after simulating the number of acquired FID samples.

    Shapes
    ------
    spectra:
        (batch_size, ..., n_timepoints)

        The function also supports an additional component axis,
        for example (batch_size, 2, n_timepoints), so that input
        and target can be transformed with exactly the same
        acquisition lengths in one batched operation.

    acquired_n_timepoints:
        (batch_size,)

        Number of actually acquired FID samples for each spectrum.
        All later FID samples were set to zero before the final FFT.
    """

    spectra: torch.Tensor
    acquired_n_timepoints: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(
            self.spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.spectra.device


def _validate_generator_device(
    *,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    """
    Ensure that random sampling happens on the same device as
    the spectra.
    """
    generator_device = torch.device(
        generator.device
    )

    if generator_device.type != device.type:
        raise ValueError(
            "Generator and spectra must use the same "
            "device type:\n"
            f"  generator: {generator_device}\n"
            f"  spectra:   {device}"
        )

    if (
        device.type == "cuda"
        and generator_device.index is not None
        and device.index is not None
        and generator_device.index != device.index
    ):
        raise ValueError(
            "Generator and spectra must use the same "
            "CUDA device:\n"
            f"  generator: {generator_device}\n"
            f"  spectra:   {device}"
        )


def _sample_acquired_n_timepoints(
    *,
    batch_size: int,
    minimum: int,
    maximum: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample one inclusive acquisition length per batch element.

    No random numbers are generated when minimum == maximum.
    """
    if minimum == maximum:
        return torch.full(
            (batch_size,),
            fill_value=minimum,
            device=device,
            dtype=torch.int64,
        )

    return torch.randint(
        low=minimum,
        high=maximum + 1,
        size=(batch_size,),
        generator=generator,
        device=device,
        dtype=torch.int64,
    )


def simulate_acquisition_length(
    *,
    spectra: torch.Tensor,
    config: SimulationConfig,
    generator: torch.Generator,
) -> AcquisitionLengthResult:
    """
    Simulate a finite acquisition length followed by zero-filling.

    Processing
    ----------
    1. Sample the number of acquired FID points for every batch item.
    2. Convert fft-shifted spectra to FIDs.
    3. Keep only the acquired FID samples.
    4. Set all later FID samples to zero.
    5. Transform back to fft-shifted spectra.

    Fast path
    ---------
    If

        min_acquired_n_timepoints
        == max_acquired_n_timepoints
        == n_timepoints

    the input tensor is returned unchanged. No IFFT, FFT, mask,
    or random-number generation is performed.

    Parameters
    ----------
    spectra:
        Complex fft-shifted spectra with shape

            (batch_size, ..., n_timepoints)

        The first dimension must be the batch dimension and the
        final dimension the spectral dimension.

    config:
        Complete simulation configuration.

    generator:
        Explicit torch random generator on the same device as
        spectra.

    Returns
    -------
    AcquisitionLengthResult
        Transformed spectra and the sampled acquisition lengths.
    """
    if spectra.ndim < 2:
        raise ValueError(
            "spectra must have shape "
            "(batch_size, ..., n_timepoints), "
            f"but found {tuple(spectra.shape)}."
        )

    if not torch.is_complex(
        spectra
    ):
        raise TypeError(
            "spectra must be complex-valued."
        )

    batch_size = int(
        spectra.shape[0]
    )

    if batch_size <= 0:
        raise ValueError(
            "spectra must contain at least one batch item."
        )

    n_timepoints = int(
        config.acquisition.n_timepoints
    )

    if int(
        spectra.shape[-1]
    ) != n_timepoints:
        raise ValueError(
            "Spectrum length does not match "
            "acquisition.n_timepoints:\n"
            f"  spectra: {int(spectra.shape[-1])}\n"
            f"  config:  {n_timepoints}"
        )

    minimum = int(
        config
        .acquisition
        .min_acquired_n_timepoints
    )

    maximum = int(
        config
        .acquisition
        .max_acquired_n_timepoints
    )

    if minimum <= 0:
        raise ValueError(
            "acquisition.min_acquired_n_timepoints "
            "must be > 0."
        )

    if maximum < minimum:
        raise ValueError(
            "acquisition.max_acquired_n_timepoints "
            "must be >= "
            "acquisition.min_acquired_n_timepoints."
        )

    if maximum > n_timepoints:
        raise ValueError(
            "acquisition.max_acquired_n_timepoints "
            "must be <= acquisition.n_timepoints."
        )

    device = spectra.device

    _validate_generator_device(
        generator=generator,
        device=device,
    )

    acquired_n_timepoints = (
        _sample_acquired_n_timepoints(
            batch_size=batch_size,
            minimum=minimum,
            maximum=maximum,
            device=device,
            generator=generator,
        )
    )

    # Exact identity configuration:
    # no random draw, no mask, no IFFT, and no FFT.
    if (
        minimum == n_timepoints
        and maximum == n_timepoints
    ):
        return AcquisitionLengthResult(
            spectra=spectra,
            acquired_n_timepoints=(
                acquired_n_timepoints
            ),
        )

    fids = torch.fft.ifft(
        torch.fft.ifftshift(
            spectra,
            dim=-1,
        ),
        dim=-1,
    )

    time_indices = torch.arange(
        n_timepoints,
        device=device,
        dtype=torch.int64,
    )

    acquisition_mask = (
        time_indices[None, :]
        < acquired_n_timepoints[:, None]
    )

    # Broadcast the same per-sample mask over any intermediate
    # component dimensions, e.g. (B, 2, T).
    mask_shape = (
        (batch_size,)
        + (1,) * (spectra.ndim - 2)
        + (n_timepoints,)
    )

    acquisition_mask = (
        acquisition_mask.view(
            mask_shape
        )
    )

    zero_filled_fids = (
        fids
        * acquisition_mask
    )

    transformed_spectra = (
        torch.fft.fftshift(
            torch.fft.fft(
                zero_filled_fids,
                dim=-1,
            ),
            dim=-1,
        )
        .contiguous()
    )

    return AcquisitionLengthResult(
        spectra=transformed_spectra,
        acquired_n_timepoints=(
            acquired_n_timepoints.contiguous()
        ),
    )
