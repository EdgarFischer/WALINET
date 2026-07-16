# src/walinet/training_data/metabolite_noise.py

from __future__ import annotations

from dataclasses import dataclass

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.metabolite_simulation import (
    SimulatedMetabolites,
)


@dataclass(frozen=True)
class SimulatedNoise:
    """
    Unscaled complex white receiver noise and sampled target SNR.

    The final noise scaling is performed after acquisition-length
    simulation so that the requested LCModel-compatible SNR is
    reproduced in the final frequency-domain spectrum.

    Shapes
    ------
    noise_spectra:
        Unscaled complex white noise, shape
        (batch_size, n_timepoints).

    snr:
        Sampled target LCModel-compatible SNR, shape
        (batch_size,).
    """

    noise_spectra: torch.Tensor
    snr: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(
            self.noise_spectra.shape[0]
        )

    @property
    def n_timepoints(self) -> int:
        return int(
            self.noise_spectra.shape[-1]
        )

    @property
    def device(self) -> torch.device:
        return self.noise_spectra.device


def _validate_generator_device(
    *,
    generator: torch.Generator,
    device: torch.device,
) -> None:
    """
    Ensure that the random generator and output tensors use
    the same device.
    """
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


def _sample_snr(
    *,
    batch_size: int,
    config: SimulationConfig,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    """
    Sample target LCModel-compatible SNR from a normal
    distribution truncated at the configured lower bound.

    Values below noise.snr.min are rejected and sampled again.
    They are not clipped, avoiding an artificial accumulation
    exactly at the lower bound.
    """
    mean = float(
        config.noise.snr.mean
    )

    std = float(
        config.noise.snr.std
    )

    minimum = float(
        config.noise.snr.min
    )

    if std == 0:
        if mean < minimum:
            raise ValueError(
                "Cannot sample SNR when noise.snr.std == 0 "
                "and noise.snr.mean < noise.snr.min."
            )

        return torch.full(
            (batch_size,),
            fill_value=mean,
            dtype=dtype,
            device=device,
        )

    snr = (
        mean
        + std
        * torch.randn(
            (batch_size,),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    )

    invalid = snr < minimum

    while torch.any(
        invalid
    ):
        n_invalid = int(
            invalid.sum().item()
        )

        snr[invalid] = (
            mean
            + std
            * torch.randn(
                (n_invalid,),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )

        invalid = snr < minimum

    return snr.contiguous()


def simulate_receiver_noise(
    *,
    metabolites: SimulatedMetabolites,
    config: SimulationConfig,
    generator: torch.Generator,
) -> SimulatedNoise:
    """
    Sample target SNR and generate unscaled complex white
    receiver noise.

    The final noise amplitude is intentionally not determined here.

    Variable acquisition length and zero-filling subsequently alter
    the frequency-domain noise realization. Therefore, the noise is
    scaled after acquisition-length simulation according to

        LCModel SNR =
            maximum real metabolite peak
            / (2 * RMS(real receiver noise)).

    This ensures that the requested SNR applies to the final
    frequency-domain spectrum used for training.
    """
    clean_spectra = (
        metabolites.clean_spectra
    )

    if clean_spectra.ndim != 2:
        raise ValueError(
            "metabolites.clean_spectra must have shape "
            "(B, T), but found "
            f"{tuple(clean_spectra.shape)}."
        )

    if not torch.is_complex(
        clean_spectra
    ):
        raise TypeError(
            "metabolites.clean_spectra must be complex-valued."
        )

    if not torch.isfinite(
        clean_spectra.real
    ).all():
        raise ValueError(
            "Clean metabolite spectra contain non-finite "
            "real values."
        )

    if not torch.isfinite(
        clean_spectra.imag
    ).all():
        raise ValueError(
            "Clean metabolite spectra contain non-finite "
            "imaginary values."
        )

    device = clean_spectra.device
    real_dtype = clean_spectra.real.dtype

    _validate_generator_device(
        generator=generator,
        device=device,
    )

    batch_size = int(
        clean_spectra.shape[0]
    )

    n_timepoints = int(
        clean_spectra.shape[-1]
    )

    snr = _sample_snr(
        batch_size=batch_size,
        config=config,
        device=device,
        dtype=real_dtype,
        generator=generator,
    )

    noise_real = torch.randn(
        (
            batch_size,
            n_timepoints,
        ),
        generator=generator,
        device=device,
        dtype=real_dtype,
    )

    noise_imag = torch.randn(
        (
            batch_size,
            n_timepoints,
        ),
        generator=generator,
        device=device,
        dtype=real_dtype,
    )

    noise_spectra = torch.complex(
        noise_real,
        noise_imag,
    ).contiguous()

    if not torch.isfinite(
        noise_spectra.real
    ).all():
        raise RuntimeError(
            "Generated noise spectra contain non-finite "
            "real values."
        )

    if not torch.isfinite(
        noise_spectra.imag
    ).all():
        raise RuntimeError(
            "Generated noise spectra contain non-finite "
            "imaginary values."
        )

    return SimulatedNoise(
        noise_spectra=noise_spectra,
        snr=snr,
    )