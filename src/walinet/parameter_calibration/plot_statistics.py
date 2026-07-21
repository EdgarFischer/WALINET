from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm


DistributionModel = Literal[
    "normal",
    "truncated_normal",
    "lognormal",
]


# For a normally distributed variable:
#
#     IQR = 1.349 * sigma
#
NORMAL_IQR_FACTOR = 1.349


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
        Model location parameter.

        For ``normal`` and ``truncated_normal`` this is the mean on the
        original scale.

        For ``lognormal`` this is ``log_mu``, the location parameter on
        the natural-logarithmic scale.

    std:
        Model scale parameter. Must be strictly positive.

        For ``normal`` and ``truncated_normal`` this is the standard
        deviation on the original scale.

        For ``lognormal`` this is ``log_sigma``, the standard deviation
        on the natural-logarithmic scale.

    distribution:
        Probability model to overlay:

        ``"normal"``
            Ordinary normal distribution.

        ``"truncated_normal"``
            Normal distribution truncated at zero. Negative samples are
            assumed to be rejected and newly sampled.

        ``"lognormal"``
            Lognormal distribution. Only strictly positive values are
            used.

    xlabel:
        Label of the x-axis, including the unit.

    ylabel:
        Label of the y-axis.

    title:
        Optional title.

    bins:
        Number of histogram bins or a NumPy binning strategy such as
        ``"auto"``, ``"fd"``, or ``"sturges"``.

    n_sigmas:
        Number of model standard deviations used for the automatic
        plotting range.

        For a normal distribution:

            x_min = mean - n_sigmas * std
            x_max = mean + n_sigmas * std

        For a zero-truncated normal distribution:

            x_min = max(0, mean - n_sigmas * std)
            x_max = mean + n_sigmas * std

        For a lognormal distribution, ``mean`` and ``std`` represent
        ``log_mu`` and ``log_sigma``:

            x_min = 0
            x_max = exp(log_mu + n_sigmas * log_sigma)

        This parameter is ignored when ``x_limits`` is provided.

    x_limits:
        Optional explicit x-axis limits as ``(minimum, maximum)``.
        When provided, these limits override ``n_sigmas``.

    save_path:
        Optional path under which the figure is saved. The parent
        directory is created automatically.

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

    if distribution not in {
        "normal",
        "truncated_normal",
        "lognormal",
    }:
        raise ValueError(
            "distribution must be 'normal', "
            "'truncated_normal', or 'lognormal'."
        )

    if distribution == "lognormal":
        values = values[
            values > 0
        ]

    if values.size == 0:
        if distribution == "lognormal":
            raise ValueError(
                "pooled_values contains no strictly positive finite "
                "values for the lognormal model."
            )

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

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    # -------------------------------------------------------------
    # Plotting limits
    # -------------------------------------------------------------
    if x_limits is None:
        if distribution == "normal":
            x_min = (
                mean
                - n_sigmas * std
            )

            x_max = (
                mean
                + n_sigmas * std
            )

        elif distribution == "truncated_normal":
            x_min = max(
                0.0,
                mean - n_sigmas * std,
            )

            x_max = (
                mean
                + n_sigmas * std
            )

        else:
            # For the lognormal model, mean and std represent
            # log_mu and log_sigma.
            x_min = 0.0

            log_x_max = (
                mean
                + n_sigmas * std
            )

            x_max = float(
                np.exp(log_x_max)
            )

        if (
            not np.isfinite(x_min)
            or not np.isfinite(x_max)
            or x_min >= x_max
        ):
            raise ValueError(
                "The automatically calculated x-axis limits are "
                "invalid.\n"
                f"  distribution: {distribution}\n"
                f"  x_min:        {x_min}\n"
                f"  x_max:        {x_max}\n"
                f"  mean:         {mean}\n"
                f"  std:          {std}\n"
                f"  n_sigmas:     {n_sigmas}"
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

        if (
            distribution == "lognormal"
            and x_min < 0
        ):
            raise ValueError(
                "The lower x-limit must not be negative for a "
                "lognormal distribution."
            )

        if (
            distribution in {
                "truncated_normal",
                "lognormal",
            }
            and x_max <= 0
        ):
            raise ValueError(
                "The upper x-limit must be greater than zero for "
                f"distribution={distribution!r}."
            )

    # -------------------------------------------------------------
    # Model density
    # -------------------------------------------------------------
    if distribution == "lognormal":
        # Do not evaluate the lognormal density exactly at zero.
        model_x_min = max(
            x_min,
            np.finfo(np.float64).tiny,
        )

        x = np.linspace(
            model_x_min,
            x_max,
            n_model_points,
        )

        model_pdf = lognorm.pdf(
            x,
            s=std,
            scale=np.exp(mean),
        )

        model_label = (
            "Lognormal model: "
            f"log-μ={mean:.4g}, "
            f"log-σ={std:.4g}"
        )

    elif distribution == "truncated_normal":
        x = np.linspace(
            x_min,
            x_max,
            n_model_points,
        )

        lower_z = (
            0.0 - mean
        ) / std

        probability_above_zero = norm.sf(
            lower_z
        )

        if (
            not np.isfinite(probability_above_zero)
            or probability_above_zero <= 0
        ):
            raise ValueError(
                "The truncated normal distribution has "
                "numerically zero probability above zero."
            )

        model_pdf = (
            norm.pdf(
                x,
                loc=mean,
                scale=std,
            )
            / probability_above_zero
        )

        model_pdf = np.where(
            x >= 0,
            model_pdf,
            0.0,
        )

        model_label = (
            "Zero-truncated normal model: "
            f"μ={mean:.4g}, "
            f"σ={std:.4g}"
        )

    else:
        x = np.linspace(
            x_min,
            x_max,
            n_model_points,
        )

        model_pdf = norm.pdf(
            x,
            loc=mean,
            scale=std,
        )

        model_label = (
            "Normal model: "
            f"μ={mean:.4g}, "
            f"σ={std:.4g}"
        )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
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


def compare_positive_models(
    values: np.ndarray,
    *,
    title: str = "Distribution comparison",
    xlabel: str = "Value",
    bins: int | str = 80,
    truncated_normal_sigma_factor: float = 2.0,
    lognormal_sigma_factor: float = 1.0,
    plot_percentile: float = 99.5,
) -> dict[str, float]:
    """
    Compare a robust zero-truncated normal model with a robust
    lognormal model for strictly positive values.

    The zero-truncated normal model is calibrated as:

        mu = median(values)
        sigma = truncated_normal_sigma_factor * IQR(values)

    The lognormal model is calibrated as:

        log_mu = median(log(values))
        robust_log_sigma = IQR(log(values)) / 1.349
        log_sigma = lognormal_sigma_factor * robust_log_sigma

    ``plot_percentile`` affects only the displayed plotting range. It
    does not affect either model's parameter estimation.
    """
    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    values = values[
        np.isfinite(values)
        & (values > 0)
    ]

    if values.size == 0:
        raise ValueError(
            "No positive finite values provided."
        )

    truncated_normal_sigma_factor = float(
        truncated_normal_sigma_factor
    )

    if (
        not np.isfinite(truncated_normal_sigma_factor)
        or truncated_normal_sigma_factor <= 0
    ):
        raise ValueError(
            "truncated_normal_sigma_factor must be finite and "
            "greater than zero."
        )

    lognormal_sigma_factor = float(
        lognormal_sigma_factor
    )

    if (
        not np.isfinite(lognormal_sigma_factor)
        or lognormal_sigma_factor <= 0
    ):
        raise ValueError(
            "lognormal_sigma_factor must be finite and "
            "greater than zero."
        )

    plot_percentile = float(
        plot_percentile
    )

    if (
        not np.isfinite(plot_percentile)
        or not 0 < plot_percentile <= 100
    ):
        raise ValueError(
            "plot_percentile must be finite and in (0, 100]."
        )

    # -------------------------------------------------------------
    # Linear-scale robust statistics
    # -------------------------------------------------------------
    q1, median, q3 = np.percentile(
        values,
        [25, 50, 75],
    )

    iqr = (
        q3 - q1
    )

    truncated_normal_mu = float(
        median
    )

    truncated_normal_sigma = float(
        truncated_normal_sigma_factor
        * iqr
    )

    # -------------------------------------------------------------
    # Log-scale robust statistics
    # -------------------------------------------------------------
    log_values = np.log(
        values
    )

    log_q1, log_mu, log_q3 = np.percentile(
        log_values,
        [25, 50, 75],
    )

    log_iqr = (
        log_q3 - log_q1
    )

    robust_log_sigma = float(
        log_iqr
        / NORMAL_IQR_FACTOR
    )

    log_sigma = float(
        lognormal_sigma_factor
        * robust_log_sigma
    )

    if (
        not np.isfinite(truncated_normal_sigma)
        or truncated_normal_sigma <= 0
    ):
        raise ValueError(
            "The linear IQR is zero or invalid."
        )

    if (
        not np.isfinite(robust_log_sigma)
        or robust_log_sigma <= 0
    ):
        raise ValueError(
            "The logarithmic IQR is zero or invalid."
        )

    if (
        not np.isfinite(log_sigma)
        or log_sigma <= 0
    ):
        raise ValueError(
            "The scaled logarithmic standard deviation is invalid."
        )

    # -------------------------------------------------------------
    # Plotting range only
    # -------------------------------------------------------------
    x_max = float(
        np.percentile(
            values,
            plot_percentile,
        )
    )

    if (
        not np.isfinite(x_max)
        or x_max <= 0
    ):
        raise ValueError(
            "The calculated plotting maximum is invalid."
        )

    x = np.linspace(
        max(
            x_max * 1e-8,
            np.finfo(np.float64).tiny,
        ),
        x_max,
        1200,
    )

    # -------------------------------------------------------------
    # Zero-truncated normal density
    # -------------------------------------------------------------
    probability_above_zero = norm.sf(
        -truncated_normal_mu
        / truncated_normal_sigma
    )

    if (
        not np.isfinite(probability_above_zero)
        or probability_above_zero <= 0
    ):
        raise ValueError(
            "The truncated normal distribution has numerically zero "
            "probability above zero."
        )

    truncated_normal_pdf = (
        norm.pdf(
            x,
            loc=truncated_normal_mu,
            scale=truncated_normal_sigma,
        )
        / probability_above_zero
    )

    # -------------------------------------------------------------
    # Lognormal density
    # -------------------------------------------------------------
    lognormal_pdf = lognorm.pdf(
        x,
        s=log_sigma,
        scale=np.exp(log_mu),
    )

    # -------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        values,
        bins=bins,
        range=(
            0,
            x_max,
        ),
        density=True,
        alpha=0.6,
        label="In-vivo values",
    )

    ax.plot(
        x,
        truncated_normal_pdf,
        linewidth=2,
        label=(
            "Zero-truncated normal\n"
            f"μ={truncated_normal_mu:.2f}, "
            f"σ={truncated_normal_sigma:.2f}"
        ),
    )

    ax.plot(
        x,
        lognormal_pdf,
        linewidth=2,
        label=(
            "Lognormal\n"
            f"log-μ={log_mu:.2f}, "
            f"log-σ={log_sigma:.2f}, "
            f"factor={lognormal_sigma_factor:.2f}"
        ),
    )

    ax.axvline(
        median,
        linestyle="--",
        linewidth=1,
        label=f"Median={median:.2f}",
    )

    ax.set_xlim(
        0,
        x_max,
    )

    ax.set_xlabel(
        xlabel
    )

    ax.set_ylabel(
        "Density"
    )

    ax.set_title(
        title
    )

    ax.legend()

    fig.tight_layout()

    plt.show()

    # -------------------------------------------------------------
    # Output
    # -------------------------------------------------------------
    print("Linear scale:")
    print(
        f"  median = {median:.4f}"
    )
    print(
        f"  IQR    = {iqr:.4f}"
    )

    print("\nTruncated normal:")
    print(
        f"  mu     = {truncated_normal_mu:.4f}"
    )
    print(
        f"  sigma  = {truncated_normal_sigma:.4f}"
    )

    print("\nLognormal:")
    print(
        f"  log_mu             = {log_mu:.4f}"
    )
    print(
        f"  robust_log_sigma   = {robust_log_sigma:.4f}"
    )
    print(
        f"  sigma_factor       = {lognormal_sigma_factor:.4f}"
    )
    print(
        f"  final_log_sigma    = {log_sigma:.4f}"
    )
    print(
        f"  median             = {np.exp(log_mu):.4f}"
    )

    print("\nPlot:")
    print(
        f"  x-axis shows values up to the "
        f"{plot_percentile}th percentile: {x_max:.4f}"
    )

    return {
        "median": float(median),
        "iqr": float(iqr),
        "normal_mu": truncated_normal_mu,
        "normal_sigma": truncated_normal_sigma,
        "log_mu": float(log_mu),
        "robust_log_sigma": robust_log_sigma,
        "log_sigma_factor": lognormal_sigma_factor,
        "log_sigma": log_sigma,
        "plot_xmax": x_max,
    }