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


LEGACY_NOISE_CALIBRATION = 0.65


@dataclass(frozen=True)
class SimulatedNoise:
    """
    Complex white receiver noise in the frequency domain.

    Shapes
    ------
    noise_spectra:
        (batch_size, n_timepoints)

    snr:
        (batch_size,)

    clean_spectrum_std:
        (batch_size,)

    noise_scale:
        (batch_size,)
    """

    noise_spectra: torch.Tensor

    snr: torch.Tensor
    clean_spectrum_std: torch.Tensor
    noise_scale: torch.Tensor

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
    Sample the legacy SNR parameter uniformly.
    """
    snr_min = float(
        config.noise.snr_min
    )

    snr_max = float(
        config.noise.snr_max
    )

    if snr_min <= 0:
        raise ValueError(
            "noise.snr_min must be > 0."
        )

    if snr_max < snr_min:
        raise ValueError(
            "noise.snr_max must be >= noise.snr_min."
        )

    if snr_min == snr_max:
        return torch.full(
            (batch_size,),
            fill_value=snr_min,
            dtype=dtype,
            device=device,
        )

    random_values = torch.rand(
        (batch_size,),
        generator=generator,
        device=device,
        dtype=dtype,
    )

    return (
        snr_min
        + (
            snr_max
            - snr_min
        )
        * random_values
    )


def _complex_population_std(
    spectra: torch.Tensor,
) -> torch.Tensor:
    """
    Match NumPy's default population standard deviation for
    complex-valued data:

        sqrt(mean(abs(x - mean(x)) ** 2))

    Returns
    -------
    torch.Tensor
        One real standard deviation per spectrum, shape (B,).
    """
    centered = (
        spectra
        - spectra.mean(
            dim=-1,
            keepdim=True,
        )
    )

    variance = torch.mean(
        torch.abs(
            centered
        ).square(),
        dim=-1,
    )

    return torch.sqrt(
        variance
    )


def simulate_receiver_noise(
    *,
    metabolites: SimulatedMetabolites,
    config: SimulationConfig,
    generator: torch.Generator,
) -> SimulatedNoise:
    """
    Generate complex white receiver noise directly in the
    frequency domain.

    The legacy WALINET scaling rule is retained:

        snr ~ Uniform(snr_min, snr_max)

        noise_scale =
            std(clean_metabolite_spectrum)
            / 0.65
            / snr

        noise =
            noise_scale
            * (
                Normal(0, 1)
                + i * Normal(0, 1)
            )

    No IFFT or additional FFT is required.

    The noise is added to the complete assembled spectrum later in
    spectrum_assembly.py.
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

    clean_spectrum_std = (
        _complex_population_std(
            clean_spectra
        )
    )

    if torch.any(
        clean_spectrum_std <= 0
    ):
        invalid_indices = torch.nonzero(
            clean_spectrum_std <= 0,
            as_tuple=False,
        ).squeeze(-1)

        raise RuntimeError(
            "At least one clean metabolite spectrum has "
            "zero standard deviation:\n"
            f"  batch indices: "
            f"{invalid_indices.detach().cpu().tolist()}"
        )

    noise_scale = (
        clean_spectrum_std
        / LEGACY_NOISE_CALIBRATION
        / snr
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

    noise_spectra = (
        torch.complex(
            noise_real,
            noise_imag,
        )
        * noise_scale[:, None]
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
        snr=snr.contiguous(),
        clean_spectrum_std=(
            clean_spectrum_std.contiguous()
        ),
        noise_scale=(
            noise_scale.contiguous()
        ),
    )