import matplotlib.pyplot as plt
import torch


def make_ppm_axis(
    n_timepoints: int,
    bandwidth_hz: float,
    nmr_frequency_hz: float,
    reference_ppm: float = 4.68,
) -> torch.Tensor:
    """
    Create an fft-shifted ppm axis.

    Parameters
    ----------
    n_timepoints:
        Number of spectral points.

    bandwidth_hz:
        Spectral bandwidth in Hz.

    nmr_frequency_hz:
        Scanner frequency in Hz.

    reference_ppm:
        Chemical-shift reference in ppm.
        For water-referenced proton spectra usually 4.68 ppm.
    """
    dwell_time = 1.0 / bandwidth_hz

    frequency_axis_hz = torch.fft.fftshift(
        torch.fft.fftfreq(
            n_timepoints,
            d=dwell_time,
        )
    )

    hz_per_ppm = (
        nmr_frequency_hz
        / 1e6
    )

    ppm_axis = (
        reference_ppm
        - frequency_axis_hz / hz_per_ppm
    )

    return ppm_axis

def plot_spectra_range(
    total_spectra: torch.Tensor,
    metabolite_spectra: torch.Tensor,
    simulation_cfg,
    start: int = 0,
    stop: int = 5,
    component: str = "real",
    reference_ppm: float = 4.68,
    ppm_min: float | None = None,
    ppm_max: float | None = None,
    figsize_per_row: float = 2.5,
):
    """
    Plot a range of spectra.

    Left:
        Complete spectrum

    Right:
        Metabolites + noise

    component:
        "real", "imag", or "abs"

    stop is exclusive.
    """
    if total_spectra.ndim != 2:
        raise ValueError(
            "total_spectra must have shape (N, T)."
        )

    if metabolite_spectra.ndim != 2:
        raise ValueError(
            "metabolite_spectra must have shape (N, T)."
        )

    if total_spectra.shape != metabolite_spectra.shape:
        raise ValueError(
            "Both tensors must have the same shape."
        )

    if component not in {
        "real",
        "imag",
        "abs",
    }:
        raise ValueError(
            "component must be 'real', 'imag', or 'abs'."
        )

    n_spectra, n_timepoints = total_spectra.shape

    if start < 0:
        raise ValueError(
            "start must be >= 0."
        )

    if stop <= start:
        raise ValueError(
            "stop must be greater than start."
        )

    if stop > n_spectra:
        raise ValueError(
            f"stop={stop} exceeds the number of spectra "
            f"({n_spectra})."
        )

    ppm_axis = make_ppm_axis(
        n_timepoints=n_timepoints,
        bandwidth_hz=(
            simulation_cfg
            .acquisition
            .bandwidth_hz
        ),
        nmr_frequency_hz=(
            simulation_cfg
            .acquisition
            .nmr_frequency_hz
        ),
        reference_ppm=reference_ppm,
    ).cpu().numpy()

    def get_component(
        spectra: torch.Tensor,
    ) -> torch.Tensor:
        if component == "real":
            return spectra.real

        if component == "imag":
            return spectra.imag

        return torch.abs(
            spectra
        )

    total_plot = get_component(
        total_spectra[start:stop]
    ).detach().cpu().numpy()

    metabolite_plot = get_component(
        metabolite_spectra[start:stop]
    ).detach().cpu().numpy()

    n_rows = stop - start

    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=2,
        figsize=(
            14,
            figsize_per_row * n_rows,
        ),
        squeeze=False,
        sharex=True,
    )

    for row, spectrum_index in enumerate(
        range(start, stop)
    ):
        left_axis = axes[row, 0]
        right_axis = axes[row, 1]

        left_axis.plot(
            ppm_axis,
            total_plot[row],
        )

        right_axis.plot(
            ppm_axis,
            metabolite_plot[row],
        )

        left_axis.axhline(
            0.0,
            linewidth=0.5,
            alpha=0.5,
        )

        right_axis.axhline(
            0.0,
            linewidth=0.5,
            alpha=0.5,
        )

        left_axis.set_ylabel(
            f"#{spectrum_index}"
        )

        left_axis.grid(
            alpha=0.2
        )

        right_axis.grid(
            alpha=0.2
        )

        if (
            ppm_min is not None
            and ppm_max is not None
        ):
            left_axis.set_xlim(
                ppm_max,
                ppm_min,
            )

            right_axis.set_xlim(
                ppm_max,
                ppm_min,
            )

        else:
            left_axis.invert_xaxis()
            right_axis.invert_xaxis()

    axes[0, 0].set_title(
        f"Complete spectrum — {component}"
    )

    axes[0, 1].set_title(
        f"Metabolites + noise — {component}"
    )

    axes[-1, 0].set_xlabel(
        "ppm"
    )

    axes[-1, 1].set_xlabel(
        "ppm"
    )

    fig.tight_layout()

    return fig, axes