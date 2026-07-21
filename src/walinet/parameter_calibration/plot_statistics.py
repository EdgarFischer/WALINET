from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm, norm


SingleDistributionModel = Literal[
    "normal",
    "truncated_normal",
    "lognormal",
]


# For a normally distributed variable:
#
#     IQR = 1.349 * sigma
#
NORMAL_IQR_FACTOR = 1.349

# Fixed global mixture weights. These are deliberately identical for
# every calibrated simulation parameter.
NORMAL_COMPONENT_WEIGHT = 0.5
LOGNORMAL_COMPONENT_WEIGHT = 0.5


ModelStatistics = dict[str, Any]


def _as_finite_1d(
    values: np.ndarray,
    *,
    positive_only: bool,
) -> np.ndarray:
    """Return finite one-dimensional values, optionally restricted to > 0."""
    values = np.asarray(
        values,
        dtype=np.float64,
    ).ravel()

    valid = np.isfinite(
        values
    )

    if positive_only:
        valid &= values > 0

    values = values[
        valid
    ]

    if values.size == 0:
        if positive_only:
            raise ValueError(
                "No strictly positive finite values were provided."
            )

        raise ValueError(
            "No finite values were provided."
        )

    return values


def _validate_plot_percentile(
    plot_percentile: float,
) -> float:
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

    return plot_percentile


def _validate_positive_factor(
    value: float,
    *,
    name: str,
) -> float:
    value = float(
        value
    )

    if (
        not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"{name} must be finite and greater than zero."
        )

    return value


def _robust_linear_parameters(
    values: np.ndarray,
    *,
    sigma_factor: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """
    Estimate a robust normal location and scale.

    Returns
    -------
    q1, median, q3, iqr, sigma

    with

        sigma = sigma_factor * IQR(values) / 1.349
    """
    q1, median, q3 = np.percentile(
        values,
        [25, 50, 75],
    )

    iqr = float(
        q3 - q1
    )

    sigma = float(
        sigma_factor
        * iqr
        / NORMAL_IQR_FACTOR
    )

    if (
        not np.isfinite(iqr)
        or iqr <= 0
    ):
        raise ValueError(
            "The linear IQR is zero or invalid."
        )

    if (
        not np.isfinite(sigma)
        or sigma <= 0
    ):
        raise ValueError(
            "The robust normal standard deviation is invalid."
        )

    return (
        float(q1),
        float(median),
        float(q3),
        iqr,
        sigma,
    )


def _robust_lognormal_parameters(
    positive_values: np.ndarray,
    *,
    sigma_factor: float = 1.0,
) -> tuple[float, float, float, float, float]:
    """
    Estimate robust lognormal parameters.

    Returns
    -------
    log_q1, log_mu, log_q3, robust_log_sigma, log_sigma

    with

        log_mu = median(log(values))
        robust_log_sigma = IQR(log(values)) / 1.349
        log_sigma = sigma_factor * robust_log_sigma
    """
    positive_values = _as_finite_1d(
        positive_values,
        positive_only=True,
    )

    log_values = np.log(
        positive_values
    )

    log_q1, log_mu, log_q3 = np.percentile(
        log_values,
        [25, 50, 75],
    )

    log_iqr = float(
        log_q3 - log_q1
    )

    robust_log_sigma = float(
        log_iqr
        / NORMAL_IQR_FACTOR
    )

    log_sigma = float(
        sigma_factor
        * robust_log_sigma
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
            "The robust lognormal standard deviation is invalid."
        )

    return (
        float(log_q1),
        float(log_mu),
        float(log_q3),
        robust_log_sigma,
        log_sigma,
    )


def _zero_truncated_normal_pdf(
    x: np.ndarray,
    *,
    mean: float,
    std: float,
) -> np.ndarray:
    """Density of N(mean, std²) conditioned on values greater than zero."""
    probability_above_zero = norm.sf(
        -mean
        / std
    )

    if (
        not np.isfinite(probability_above_zero)
        or probability_above_zero <= 0
    ):
        raise ValueError(
            "The truncated normal distribution has numerically zero "
            "probability above zero."
        )

    density = (
        norm.pdf(
            x,
            loc=mean,
            scale=std,
        )
        / probability_above_zero
    )

    return np.where(
        x >= 0,
        density,
        0.0,
    )


def _save_figure(
    fig: plt.Figure,
    *,
    save_path: str | Path | None,
    dpi: int,
) -> None:
    if save_path is None:
        return

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


def plot_pooled_histogram_with_model(
    pooled_values: np.ndarray,
    *,
    mean: float,
    std: float,
    distribution: SingleDistributionModel = "normal",
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
    Legacy helper for plotting a single normal, zero-truncated-normal,
    or lognormal model.

    New simulation calibration should normally use
    ``compare_positive_models`` or ``compare_symmetric_models`` below.
    """
    if distribution not in {
        "normal",
        "truncated_normal",
        "lognormal",
    }:
        raise ValueError(
            "distribution must be 'normal', 'truncated_normal', "
            "or 'lognormal'."
        )

    values = _as_finite_1d(
        pooled_values,
        positive_only=(distribution == "lognormal"),
    )

    mean = float(
        mean
    )
    std = _validate_positive_factor(
        std,
        name="std",
    )
    n_sigmas = _validate_positive_factor(
        n_sigmas,
        name="n_sigmas",
    )

    if not np.isfinite(mean):
        raise ValueError(
            "mean must be finite."
        )

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    if x_limits is None:
        if distribution == "normal":
            x_min = mean - n_sigmas * std
            x_max = mean + n_sigmas * std
        elif distribution == "truncated_normal":
            x_min = max(
                0.0,
                mean - n_sigmas * std,
            )
            x_max = mean + n_sigmas * std
        else:
            x_min = 0.0
            x_max = float(
                np.exp(
                    mean
                    + n_sigmas * std
                )
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
            "The plotting limits are invalid."
        )

    if distribution in {
        "truncated_normal",
        "lognormal",
    } and x_max <= 0:
        raise ValueError(
            "The upper plotting limit must be greater than zero."
        )

    if distribution == "lognormal":
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
            f"log-μ={mean:.4g}, log-σ={std:.4g}"
        )
    else:
        x = np.linspace(
            x_min,
            x_max,
            n_model_points,
        )

        if distribution == "truncated_normal":
            model_pdf = _zero_truncated_normal_pdf(
                x,
                mean=mean,
                std=std,
            )
            model_label = (
                "Zero-truncated normal model: "
                f"μ={mean:.4g}, σ={std:.4g}"
            )
        else:
            model_pdf = norm.pdf(
                x,
                loc=mean,
                scale=std,
            )
            model_label = (
                "Normal model: "
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
        alpha=0.6,
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

    _save_figure(
        fig,
        save_path=save_path,
        dpi=dpi,
    )

    return fig, ax


def compare_positive_models(
    values: np.ndarray,
    *,
    title: str = "Positive distribution mixture",
    xlabel: str = "Value",
    ylabel: str = "Probability density",
    bins: int | str = 80,
    truncated_normal_sigma_factor: float = 1.0,
    lognormal_sigma_factor: float = 1.0,
    plot_percentile: float = 99.5,
    x_limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1200,
    show: bool = True,
) -> ModelStatistics:
    """
    Calibrate and visualize the common model for positive parameters:

        0.5 * ZeroTruncatedNormal
        +
        0.5 * LogNormal

    Both components are calibrated robustly and directly from the data:

        normal_mu = median(values)
        normal_sigma = IQR(values) / 1.349

        log_mu = median(log(values))
        log_sigma = IQR(log(values)) / 1.349

    The two sigma-factor arguments are retained for backwards-compatible
    diagnostics. Set both to 1.0 for the final data-calibrated model.
    """
    values = _as_finite_1d(
        values,
        positive_only=True,
    )

    truncated_normal_sigma_factor = _validate_positive_factor(
        truncated_normal_sigma_factor,
        name="truncated_normal_sigma_factor",
    )
    lognormal_sigma_factor = _validate_positive_factor(
        lognormal_sigma_factor,
        name="lognormal_sigma_factor",
    )
    plot_percentile = _validate_plot_percentile(
        plot_percentile
    )

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    q1, median, q3, iqr, normal_sigma = (
        _robust_linear_parameters(
            values,
            sigma_factor=truncated_normal_sigma_factor,
        )
    )
    normal_mu = median

    (
        log_q1,
        log_mu,
        log_q3,
        robust_log_sigma,
        log_sigma,
    ) = _robust_lognormal_parameters(
        values,
        sigma_factor=lognormal_sigma_factor,
    )

    if x_limits is None:
        x_min = 0.0
        x_max = float(
            np.percentile(
                values,
                plot_percentile,
            )
        )
    else:
        x_min, x_max = map(
            float,
            x_limits,
        )

    if (
        not np.isfinite(x_min)
        or not np.isfinite(x_max)
        or x_min < 0
        or x_min >= x_max
    ):
        raise ValueError(
            "Positive-mixture x_limits must be finite and satisfy "
            "0 <= minimum < maximum."
        )

    model_x_min = max(
        x_min,
        x_max * 1e-8,
        np.finfo(np.float64).tiny,
    )

    x = np.linspace(
        model_x_min,
        x_max,
        n_model_points,
    )

    normal_pdf = _zero_truncated_normal_pdf(
        x,
        mean=normal_mu,
        std=normal_sigma,
    )

    lognormal_pdf = lognorm.pdf(
        x,
        s=log_sigma,
        scale=np.exp(log_mu),
    )

    mixture_pdf = (
        NORMAL_COMPONENT_WEIGHT
        * normal_pdf
        + LOGNORMAL_COMPONENT_WEIGHT
        * lognormal_pdf
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        values,
        bins=bins,
        range=(
            x_min,
            x_max,
        ),
        density=True,
        alpha=0.6,
        label="In-vivo values",
    )

    ax.plot(
        x,
        normal_pdf,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Zero-truncated normal component\n"
            f"μ={normal_mu:.3g}, σ={normal_sigma:.3g}"
        ),
    )

    ax.plot(
        x,
        lognormal_pdf,
        linestyle=":",
        linewidth=1.8,
        label=(
            "Lognormal component\n"
            f"log-μ={log_mu:.3g}, log-σ={log_sigma:.3g}"
        ),
    )

    ax.plot(
        x,
        mixture_pdf,
        linewidth=3,
        label="Final 50/50 mixture",
    )

    ax.axvline(
        median,
        linestyle="--",
        linewidth=1,
        label=f"Median={median:.3g}",
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
    ax.set_title(
        title
    )
    ax.legend()
    fig.tight_layout()

    _save_figure(
        fig,
        save_path=save_path,
        dpi=dpi,
    )

    if show:
        plt.show()

    print("Linear scale:")
    print(
        f"  q1                 = {q1:.6f}"
    )
    print(
        f"  median             = {median:.6f}"
    )
    print(
        f"  q3                 = {q3:.6f}"
    )
    print(
        f"  IQR                = {iqr:.6f}"
    )

    print("\nZero-truncated normal component:")
    print(
        f"  mu                 = {normal_mu:.6f}"
    )
    print(
        f"  sigma              = {normal_sigma:.6f}"
    )
    print(
        f"  sigma_factor       = "
        f"{truncated_normal_sigma_factor:.6f}"
    )

    print("\nLognormal component:")
    print(
        f"  log_q1             = {log_q1:.6f}"
    )
    print(
        f"  log_mu             = {log_mu:.6f}"
    )
    print(
        f"  log_q3             = {log_q3:.6f}"
    )
    print(
        f"  robust_log_sigma   = {robust_log_sigma:.6f}"
    )
    print(
        f"  final_log_sigma    = {log_sigma:.6f}"
    )
    print(
        f"  sigma_factor       = {lognormal_sigma_factor:.6f}"
    )

    print("\nMixture:")
    print(
        f"  normal_weight      = {NORMAL_COMPONENT_WEIGHT:.6f}"
    )
    print(
        f"  lognormal_weight   = {LOGNORMAL_COMPONENT_WEIGHT:.6f}"
    )

    return {
        "median": float(median),
        "iqr": float(iqr),
        "normal_mu": float(normal_mu),
        "normal_sigma": float(normal_sigma),
        "normal_sigma_factor": float(
            truncated_normal_sigma_factor
        ),
        "log_mu": float(log_mu),
        "robust_log_sigma": float(robust_log_sigma),
        "log_sigma_factor": float(
            lognormal_sigma_factor
        ),
        "log_sigma": float(log_sigma),
        "normal_weight": float(NORMAL_COMPONENT_WEIGHT),
        "lognormal_weight": float(LOGNORMAL_COMPONENT_WEIGHT),
        "plot_xmin": float(x_min),
        "plot_xmax": float(x_max),
        "figure": fig,
        "axes": ax,
    }


def compare_symmetric_models(
    values: np.ndarray,
    *,
    title: str = "Symmetric distribution mixture",
    xlabel: str = "Value",
    ylabel: str = "Probability density",
    bins: int | str = 80,
    plot_percentile: float = 99.5,
    x_limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1200,
    show: bool = True,
) -> ModelStatistics:
    """
    Calibrate and visualize the common model for signed parameters.

    Let ``center = median(values)`` and ``D = abs(values - center)``.
    The final model is:

        X = center + epsilon

    where

        epsilon ~ Normal(0, normal_sigma)               with p = 0.5
        epsilon ~ S * LogNormal(log_mu, log_sigma)      with p = 0.5
        P(S = -1) = P(S = +1) = 0.5

    Equivalently, the complete mixture contains:

        0.50 * central normal component
        0.25 * positive lognormal tail
        0.25 * negative lognormal tail

    Calibration is robust and data-driven:

        normal_sigma = IQR(values) / 1.349
        log_mu = median(log(abs(values - center)))
        log_sigma = IQR(log(abs(values - center))) / 1.349
    """
    values = _as_finite_1d(
        values,
        positive_only=False,
    )

    plot_percentile = _validate_plot_percentile(
        plot_percentile
    )

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    q1, center, q3, iqr, normal_sigma = (
        _robust_linear_parameters(
            values,
            sigma_factor=1.0,
        )
    )

    absolute_deviations = np.abs(
        values
        - center
    )

    positive_deviations = absolute_deviations[
        np.isfinite(absolute_deviations)
        & (absolute_deviations > 0)
    ]

    if positive_deviations.size == 0:
        raise ValueError(
            "All values are identical to the median; a lognormal "
            "tail cannot be calibrated."
        )

    (
        log_q1,
        log_mu,
        log_q3,
        robust_log_sigma,
        log_sigma,
    ) = _robust_lognormal_parameters(
        positive_deviations,
        sigma_factor=1.0,
    )

    if x_limits is None:
        maximum_deviation = float(
            np.percentile(
                absolute_deviations,
                plot_percentile,
            )
        )

        x_min = float(
            center
            - maximum_deviation
        )
        x_max = float(
            center
            + maximum_deviation
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
            "Symmetric-mixture x_limits must be finite and satisfy "
            "minimum < maximum."
        )

    x = np.linspace(
        x_min,
        x_max,
        n_model_points,
    )

    normal_pdf = norm.pdf(
        x,
        loc=center,
        scale=normal_sigma,
    )

    distances = np.abs(
        x
        - center
    )

    # This is one normalized symmetric tail density. The factor 0.5
    # splits its probability equally between the positive and negative
    # sides of the center.
    symmetric_lognormal_pdf = (
        0.5
        * lognorm.pdf(
            distances,
            s=log_sigma,
            scale=np.exp(log_mu),
        )
    )

    mixture_pdf = (
        NORMAL_COMPONENT_WEIGHT
        * normal_pdf
        + LOGNORMAL_COMPONENT_WEIGHT
        * symmetric_lognormal_pdf
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.hist(
        values,
        bins=bins,
        range=(
            x_min,
            x_max,
        ),
        density=True,
        alpha=0.6,
        label="In-vivo values",
    )

    ax.plot(
        x,
        normal_pdf,
        linestyle="--",
        linewidth=1.5,
        label=(
            "Central normal component\n"
            f"μ={center:.3g}, σ={normal_sigma:.3g}"
        ),
    )

    ax.plot(
        x,
        symmetric_lognormal_pdf,
        linestyle=":",
        linewidth=1.8,
        label=(
            "Symmetric lognormal-tail component\n"
            f"log-μ={log_mu:.3g}, log-σ={log_sigma:.3g}"
        ),
    )

    ax.plot(
        x,
        mixture_pdf,
        linewidth=3,
        label="Final 50/25/25 mixture",
    )

    ax.axvline(
        center,
        linestyle="--",
        linewidth=1,
        label=f"Center={center:.3g}",
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
    ax.set_title(
        title
    )
    ax.legend()
    fig.tight_layout()

    _save_figure(
        fig,
        save_path=save_path,
        dpi=dpi,
    )

    if show:
        plt.show()

    print("Linear scale:")
    print(
        f"  q1                 = {q1:.6f}"
    )
    print(
        f"  center / median    = {center:.6f}"
    )
    print(
        f"  q3                 = {q3:.6f}"
    )
    print(
        f"  IQR                = {iqr:.6f}"
    )

    print("\nCentral normal component:")
    print(
        f"  mu                 = {center:.6f}"
    )
    print(
        f"  sigma              = {normal_sigma:.6f}"
    )

    print("\nSymmetric lognormal-tail component:")
    print(
        f"  log_q1             = {log_q1:.6f}"
    )
    print(
        f"  log_mu             = {log_mu:.6f}"
    )
    print(
        f"  log_q3             = {log_q3:.6f}"
    )
    print(
        f"  log_sigma          = {log_sigma:.6f}"
    )
    print(
        f"  tail median        = {np.exp(log_mu):.6f}"
    )

    print("\nMixture:")
    print(
        f"  normal_weight      = {NORMAL_COMPONENT_WEIGHT:.6f}"
    )
    print(
        "  positive_tail      = 0.250000"
    )
    print(
        "  negative_tail      = 0.250000"
    )

    return {
        "median": float(center),
        "center": float(center),
        "iqr": float(iqr),
        "normal_mu": float(center),
        "normal_sigma": float(normal_sigma),
        "log_mu": float(log_mu),
        "robust_log_sigma": float(robust_log_sigma),
        "log_sigma_factor": 1.0,
        "log_sigma": float(log_sigma),
        "normal_weight": float(NORMAL_COMPONENT_WEIGHT),
        "lognormal_weight": float(LOGNORMAL_COMPONENT_WEIGHT),
        "positive_tail_weight": 0.25,
        "negative_tail_weight": 0.25,
        "plot_xmin": float(x_min),
        "plot_xmax": float(x_max),
        "figure": fig,
        "axes": ax,
    }