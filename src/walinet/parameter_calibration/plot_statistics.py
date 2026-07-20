from __future__ import annotations

from math import erf, pi, sqrt
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


DistributionModel = Literal[
    "normal",
    "truncated_normal",
]


def plot_pooled_histogram_with_model(
    pooled_values: np.ndarray,
    *,
    mean: float,
    std: float,
    distribution: DistributionModel = "normal",
    xlabel: str,
    ylabel: str = "Probability density",
    title: str | None = None,
    bins: int | str = 50,
    n_sigmas: float = 4.0,
    x_limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1000,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot a density-normalized histogram of pooled voxel values and
    overlay a probability-density model.

    Parameters
    ----------
    pooled_values:
        One-dimensional array containing pooled values from all
        subjects.

    mean:
        Mean of the overlaid normal distribution.

    std:
        Standard deviation of the overlaid normal distribution.
        Must be strictly positive.

    distribution:
        Model to overlay:

        "normal"
            Ordinary normal distribution.

        "truncated_normal"
            Normal distribution truncated at zero. Negative samples
            are assumed to be rejected and newly sampled.

    xlabel:
        Label of the x-axis, including the unit.

    ylabel:
        Label of the y-axis.

    title:
        Optional title.

    bins:
        Number of histogram bins or a NumPy binning strategy such as
        "auto", "fd", or "sturges".

    n_sigmas:
        Number of standard deviations shown on each side of the mean.

        For a normal distribution:

            x_min = mean - n_sigmas * std
            x_max = mean + n_sigmas * std

        For a zero-truncated normal distribution, x_min is additionally
        restricted to be at least zero.

        This parameter is ignored when x_limits is provided.

    x_limits:
        Optional explicit x-axis limits as (minimum, maximum).
        When provided, these limits override n_sigmas.

    save_path:
        Optional path under which the figure is saved.

        Examples:
            "figures/fwhm_histogram.png"
            Path("/data/results/fwhm_histogram.pdf")

        The parent directory is created automatically.

    dpi:
        Resolution used when saving raster formats such as PNG.

    n_model_points:
        Number of points used to draw the model density.

    Returns
    -------
    fig:
        Matplotlib figure.

    ax:
        Matplotlib axes.
    """
    values = np.asarray(
        pooled_values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "pooled_values must be one-dimensional, "
            f"but found shape {values.shape}."
        )

    values = values[
        np.isfinite(values)
    ]

    if values.size == 0:
        raise ValueError(
            "pooled_values contains no finite values."
        )

    mean = float(mean)
    std = float(std)
    n_sigmas = float(n_sigmas)

    if not np.isfinite(mean):
        raise ValueError(
            "mean must be finite."
        )

    if not np.isfinite(std) or std <= 0:
        raise ValueError(
            "std must be finite and greater than zero."
        )

    if not np.isfinite(n_sigmas) or n_sigmas <= 0:
        raise ValueError(
            "n_sigmas must be finite and greater than zero."
        )

    if distribution not in {
        "normal",
        "truncated_normal",
    }:
        raise ValueError(
            "distribution must be either "
            "'normal' or 'truncated_normal'."
        )

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    if x_limits is None:
        x_min = (
            mean
            - n_sigmas * std
        )

        x_max = (
            mean
            + n_sigmas * std
        )

        if distribution == "truncated_normal":
            x_min = max(
                0.0,
                x_min,
            )

        if x_min >= x_max:
            raise ValueError(
                "The automatically calculated x-axis limits are "
                "invalid. Check mean, std, and n_sigmas."
            )

    else:
        x_min, x_max = map(
            float,
            x_limits,
        )

        if (
            not np.isfinite(x_min)
            or not np.isfinite(x_max)
            or x_min >= x_max
        ):
            raise ValueError(
                "x_limits must contain two finite values "
                "with minimum < maximum."
            )

    x = np.linspace(
        x_min,
        x_max,
        n_model_points,
    )

    z = (
        x - mean
    ) / std

    normal_pdf = (
        np.exp(
            -0.5 * z**2
        )
        / (
            std * sqrt(2.0 * pi)
        )
    )

    if distribution == "normal":
        model_pdf = normal_pdf

        model_label = (
            f"Normal model: μ={mean:.4g}, σ={std:.4g}"
        )

    else:
        lower_z = (
            0.0 - mean
        ) / std

        probability_above_zero = (
            1.0
            - 0.5
            * (
                1.0
                + erf(
                    lower_z / sqrt(2.0)
                )
            )
        )

        if probability_above_zero <= 0:
            raise ValueError(
                "The truncated normal distribution has "
                "numerically zero probability above zero."
            )

        model_pdf = np.where(
            x >= 0.0,
            normal_pdf / probability_above_zero,
            0.0,
        )

        model_label = (
            "Zero-truncated normal model: "
            f"μ={mean:.4g}, σ={std:.4g}"
        )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.hist(
        values,
        bins=bins,
        density=True,
        range=(
            x_min,
            x_max,
        ),
        label="Pooled voxel values",
    )

    ax.plot(
        x,
        model_pdf,
        linewidth=2,
        label=model_label,
    )

    ax.set_xlim(
        x_min,
        x_max,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        ylabel
    )

    if title is not None:
        ax.set_title(
            title
        )

    ax.legend()

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=dpi,
            bbox_inches="tight",
        )

    return fig, ax