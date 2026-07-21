# src/walinet/training_data/metabolite_noise.py
# Distribution refactor: mixture_v2

from __future__ import annotations

from dataclasses import dataclass

import torch

from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.distributions import (
    sample_distribution,
    validate_generator_device,
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

    validate_generator_device(
        generator=generator,
        device=device,
    )

    batch_size = int(
        clean_spectra.shape[0]
    )

    n_timepoints = int(
        clean_spectra.shape[-1]
    )

    snr = sample_distribution(
        distribution=config.noise.snr.distribution,
        shape=(batch_size,),
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

    if not torch.isfinite(
        snr
    ).all():
        raise RuntimeError(
            "Sampled SNR contains non-finite values."
        )

    if torch.any(
        snr
        < float(
            config.noise.snr.minimum
        )
    ):
        raise RuntimeError(
            "Sampled SNR contains values below noise.snr.minimum."
        )

    return SimulatedNoise(
        noise_spectra=noise_spectra,
        snr=snr,
    )