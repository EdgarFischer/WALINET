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


def plot_simulation_model_comparison(
    total_spectra,
    nuisance_ground_truth,
    metabolite_predictions,
    simulation_cfg,
    start: int = 0,
    stop: int = 10,
    component: str = "real",
    reference_ppm: float = 4.68,
    ppm_min: float | None = 0.0,
    ppm_max: float | None = 7.0,
    figsize_per_row: float = 1.45,
):
    """Compare nuisance removal models on simulated spectra.

    Each row represents one simulated spectrum. The left panel shows the
    complete input, nuisance ground truth, and nuisance predicted by each
    model. The center panel compares predicted metabolites with ground truth.
    The right panel shows the metabolite residual ``prediction - ground truth``.

    Parameters
    ----------
    total_spectra, nuisance_ground_truth:
        Complex arrays or tensors with shape ``(N, T)``.
    metabolite_predictions:
        Mapping from model label to predicted clean/metabolite spectra with
        shape ``(N, T)``. Predicted nuisance is calculated as input minus the
        corresponding metabolite prediction.
    start, stop:
        Half-open simulation index range ``[start, stop)`` to display.
    """
    import numpy as np

    def as_numpy(values):
        if isinstance(values, torch.Tensor):
            return values.detach().cpu().numpy()
        return np.asarray(values)

    total = as_numpy(total_spectra)
    nuisance_gt = as_numpy(nuisance_ground_truth)

    if total.ndim != 2:
        raise ValueError(f"total_spectra must have shape (N, T), got {total.shape}.")
    if nuisance_gt.shape != total.shape:
        raise ValueError(
            "nuisance_ground_truth must have the same shape as total_spectra."
        )
    if not metabolite_predictions:
        raise ValueError("metabolite_predictions must contain at least one model.")
    if not (0 <= start < stop <= total.shape[0]):
        raise ValueError(
            f"Expected 0 <= start < stop <= {total.shape[0]}, got "
            f"start={start}, stop={stop}."
        )
    if component not in {"real", "imag", "abs"}:
        raise ValueError("component must be 'real', 'imag', or 'abs'.")

    predictions = {}
    for label, values in metabolite_predictions.items():
        prediction = as_numpy(values)
        if prediction.shape != total.shape:
            raise ValueError(
                f"Prediction {label!r} has shape {prediction.shape}; "
                f"expected {total.shape}."
            )
        predictions[str(label)] = prediction

    def select_component(values):
        if component == "real":
            return values.real
        if component == "imag":
            return values.imag
        return np.abs(values)

    n_timepoints = total.shape[-1]
    bandwidth_hz = float(simulation_cfg.acquisition.bandwidth_hz)
    nmr_frequency_hz = float(simulation_cfg.acquisition.nmr_frequency_hz)
    frequency_hz = np.fft.fftshift(
        np.fft.fftfreq(n_timepoints, d=1.0 / bandwidth_hz)
    )
    ppm_axis = reference_ppm - frequency_hz / (nmr_frequency_hz / 1e6)

    if ppm_min is None:
        ppm_min = float(ppm_axis.min())
    if ppm_max is None:
        ppm_max = float(ppm_axis.max())
    ppm_low, ppm_high = sorted((ppm_min, ppm_max))
    ppm_mask = (ppm_axis >= ppm_low) & (ppm_axis <= ppm_high)
    if not np.any(ppm_mask):
        raise ValueError(f"No spectral points in ppm range {ppm_min} to {ppm_max}.")

    x = ppm_axis[ppm_mask]
    metabolite_gt = total - nuisance_gt
    colors = [f"C{i}" for i in range(3, 10)] + ["C0", "C1", "C2"]
    n_rows = stop - start
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11, max(figsize_per_row * n_rows, 3.6)),
        squeeze=False,
        sharex=False,
    )

    for row, spectrum_index in enumerate(range(start, stop)):
        nuisance_axis, metabolite_axis, residual_axis = axes[row]
        nuisance_axis.plot(
            x,
            select_component(total[spectrum_index, ppm_mask]),
            color="0.65",
            linewidth=0.9,
            label="full spectrum",
        )
        nuisance_axis.plot(
            x,
            select_component(nuisance_gt[spectrum_index, ppm_mask]),
            color="black",
            linewidth=1.0,
            linestyle=":",
            label="nuisance GT",
        )
        metabolite_axis.plot(
            x,
            select_component(metabolite_gt[spectrum_index, ppm_mask]),
            color="black",
            linewidth=1.2,
            label="metabolites GT",
        )

        for color, (label, prediction) in zip(colors, predictions.items()):
            linestyle = "-" if color == "C3" else "--"
            predicted_nuisance = total - prediction
            nuisance_axis.plot(
                x,
                select_component(predicted_nuisance[spectrum_index, ppm_mask]),
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=f"{label} nuisance",
            )
            metabolite_axis.plot(
                x,
                select_component(prediction[spectrum_index, ppm_mask]),
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=f"{label} metabolites",
            )
            residual_axis.plot(
                x,
                select_component(
                    prediction[spectrum_index, ppm_mask]
                    - metabolite_gt[spectrum_index, ppm_mask]
                ),
                color=color,
                linewidth=1.0,
                linestyle=linestyle,
                label=f"{label} residual",
            )

        # Use exactly the same y scale for the metabolite comparison and
        # its residual so their amplitudes can be compared directly.
        metabolite_ylim = metabolite_axis.get_ylim()
        residual_ylim = residual_axis.get_ylim()
        common_ylim = (
            min(metabolite_ylim[0], residual_ylim[0]),
            max(metabolite_ylim[1], residual_ylim[1]),
        )
        metabolite_axis.set_ylim(common_ylim)
        residual_axis.set_ylim(common_ylim)

        nuisance_axis.set_ylabel(f"#{spectrum_index}", fontsize=7)
        for axis in (nuisance_axis, metabolite_axis, residual_axis):
            axis.set_xlim(ppm_high, ppm_low)
            axis.axhline(0.0, color="0.5", linewidth=0.5, alpha=0.5)
            axis.grid(alpha=0.15)
            axis.tick_params(axis="both", labelsize=6)
            axis.legend(loc="upper left", fontsize=5.5, framealpha=0.85)

    axes[0, 0].set_title(
        f"Full spectrum and nuisance — {component}",
        fontsize=9,
    )
    axes[0, 1].set_title(
        f"Metabolites vs. ground truth — {component}",
        fontsize=9,
    )
    axes[0, 2].set_title(
        f"Metabolite residual (prediction − GT) — {component}",
        fontsize=9,
    )
    axes[-1, 0].set_xlabel("ppm")
    axes[-1, 1].set_xlabel("ppm")
    axes[-1, 2].set_xlabel("ppm")
    fig.tight_layout(pad=0.45, w_pad=0.5, h_pad=0.3)
    return fig, axes
