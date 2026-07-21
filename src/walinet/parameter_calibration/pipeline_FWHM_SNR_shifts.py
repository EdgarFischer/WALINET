from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from walinet.parameter_calibration.compute_statistics import (
    calculate_pooled_median_iqr,
    extract_valid_voxels,
)
from walinet.parameter_calibration.load_data import (
    load_subject_maps,
)
from walinet.parameter_calibration.plot_statistics import (
    plot_pooled_histogram_with_model,
)


DistributionModel = Literal[
    "normal",
    "truncated_normal",
    "lognormal",
]

ParameterSpace = Literal[
    "linear",
    "log",
]

# For a normally distributed variable:
# IQR = 1.349 * sigma
NORMAL_IQR_FACTOR = 1.349


@dataclass(frozen=True)
class ParameterCalibrationResult:
    """
    Results produced by the parameter-calibration pipeline.

    For normal and truncated-normal models:

        model_mean = linear-space mean parameter
        model_std  = linear-space standard deviation

    For lognormal models:

        model_mean = log_mu
        model_std  = log_sigma
    """

    maps: np.ndarray
    subject_values: dict[str, np.ndarray]
    pooled_values: np.ndarray

    median: float
    iqr: float

    model_mean: float
    model_std: float
    model_parameter_space: ParameterSpace

    figure: plt.Figure
    axes: plt.Axes


def calibrate_parameter_from_maps(
    *,
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    quality_mask: np.ndarray,
    parameter_name: str,
    unit: str,
    extension: str = ".nii.gz",
    distribution: DistributionModel = "normal",
    std_iqr_factor: float = 2.0,
    xlabel: str | None = None,
    ylabel: str = "Probability density",
    title: str | None = None,
    bins: int | str = 50,
    n_sigmas: float = 4.0,
    x_limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1000,
    print_decimals: int = 4,
    show: bool = True,
) -> ParameterCalibrationResult:
    """
    Load parameter maps, extract valid voxels, calculate pooled
    statistics, print the results, and plot the pooled distribution.

    Model calibration
    -----------------
    For ``distribution="normal"`` and
    ``distribution="truncated_normal"``:

        model_mean = median(values)
        model_std  = std_iqr_factor * IQR(values)

    For ``distribution="lognormal"``:

        log_values = log(values)
        model_mean = median(log_values)
        model_std  = IQR(log_values) / 1.349

    In the lognormal case, ``model_mean`` and ``model_std`` therefore
    correspond to ``log_mu`` and ``log_sigma``. Only finite, strictly
    positive values are used.

    Parameters
    ----------
    base_paths:
        Subject directories passed to ``load_subject_maps``.

    relative_path:
        Relative path of the parameter map within each subject
        directory.

    quality_mask:
        Binary quality mask with the same shape as the loaded maps.

    parameter_name:
        Human-readable parameter name used in printed output.

    unit:
        Unit shown in the printed output and axis label.

    extension:
        File extension passed to ``load_subject_maps``.

    distribution:
        Probability model overlaid on the histogram.

        Supported values:

            "normal"
            "truncated_normal"
            "lognormal"

    std_iqr_factor:
        Factor used for normal and truncated-normal models:

            model_std = std_iqr_factor * IQR

        This argument is ignored for a lognormal model because its
        scale is calculated as:

            log_sigma = IQR(log(values)) / 1.349

    xlabel:
        Optional custom x-axis label.

    ylabel:
        Label of the y-axis.

    title:
        Optional plot title.

    bins:
        Number of histogram bins or NumPy binning strategy.

    n_sigmas:
        Number of model standard deviations shown around the model
        location. For a lognormal model, the corresponding range is
        calculated in log space.

    x_limits:
        Optional explicit x-axis limits.

    save_path:
        Optional path under which the plot is saved.

    dpi:
        Resolution used when saving raster images.

    n_model_points:
        Number of points used to draw the model density.

    print_decimals:
        Number of decimal places used for printed values.

    show:
        Whether to call ``plt.show()``.

    Returns
    -------
    ParameterCalibrationResult:
        Object containing maps, extracted values, statistics, model
        parameters, figure, and axes.
    """
    if not parameter_name.strip():
        raise ValueError(
            "parameter_name must not be empty."
        )

    if not unit.strip():
        raise ValueError(
            "unit must not be empty."
        )

    valid_distributions = {
        "normal",
        "truncated_normal",
        "lognormal",
    }

    if distribution not in valid_distributions:
        raise ValueError(
            f"Unsupported distribution: {distribution!r}. "
            f"Supported values are: "
            f"{sorted(valid_distributions)}."
        )

    if print_decimals < 0:
        raise ValueError(
            "print_decimals must be greater than or equal to zero."
        )

    # std_iqr_factor is only used for normal models.
    if distribution != "lognormal":
        std_iqr_factor = float(
            std_iqr_factor
        )

        if (
            not np.isfinite(std_iqr_factor)
            or std_iqr_factor <= 0
        ):
            raise ValueError(
                "std_iqr_factor must be finite and greater than zero."
            )

    maps = load_subject_maps(
        base_paths=base_paths,
        relative_path=relative_path,
        extension=extension,
    )

    subject_values, pooled_values = extract_valid_voxels(
        maps=maps,
        quality_mask=quality_mask,
    )

    pooled_values = np.asarray(
        pooled_values,
        dtype=np.float64,
    )

    # A lognormal model requires strictly positive values.
    if distribution == "lognormal":
        subject_values = {
            subject: np.asarray(
                values,
                dtype=np.float64,
            )[
                np.isfinite(values)
                & (np.asarray(values) > 0)
            ]
            for subject, values in subject_values.items()
        }

        positive_mask = (
            np.isfinite(pooled_values)
            & (pooled_values > 0)
        )

        number_removed = int(
            pooled_values.size
            - np.count_nonzero(positive_mask)
        )

        pooled_values = pooled_values[
            positive_mask
        ]

        if pooled_values.size == 0:
            raise ValueError(
                "No positive finite pooled values are available "
                "for lognormal calibration."
            )

        if number_removed > 0:
            print(
                f"Ignored {number_removed} non-positive or "
                "non-finite values for lognormal calibration."
            )

    # Statistics on the original linear scale.
    median, iqr = calculate_pooled_median_iqr(
        pooled_values
    )

    median = float(median)
    iqr = float(iqr)

    if not np.isfinite(iqr) or iqr <= 0:
        raise ValueError(
            "The pooled IQR must be finite and greater than zero.\n"
            f"  median: {median}\n"
            f"  IQR:    {iqr}"
        )

    if distribution == "lognormal":
        # Robust lognormal calibration.
        log_values = np.log(
            pooled_values
        )

        log_mu, log_iqr = calculate_pooled_median_iqr(
            log_values
        )

        model_mean = float(log_mu)
        model_std = float(
            log_iqr / NORMAL_IQR_FACTOR
        )
        model_parameter_space: ParameterSpace = "log"

    else:
        # Deliberately broad robust normal calibration.
        model_mean = median
        model_std = float(
            std_iqr_factor * iqr
        )
        model_parameter_space = "linear"

    if (
        not np.isfinite(model_mean)
        or not np.isfinite(model_std)
        or model_std <= 0
    ):
        raise ValueError(
            "Invalid model parameters were calculated.\n"
            f"  distribution: {distribution}\n"
            f"  model_mean:   {model_mean}\n"
            f"  model_std:    {model_std}"
        )

    print(
        f"{parameter_name}:"
    )
    print(
        f"Median: {median:.{print_decimals}f} {unit}"
    )
    print(
        f"IQR:    {iqr:.{print_decimals}f} {unit}"
    )

    if distribution == "lognormal":
        print(
            f"Model log-μ: {model_mean:.{print_decimals}f}"
        )
        print(
            f"Model log-σ: {model_std:.{print_decimals}f}"
        )
        print(
            f"Model median: "
            f"{np.exp(model_mean):.{print_decimals}f} {unit}"
        )
    else:
        print(
            f"Model μ: {model_mean:.{print_decimals}f} {unit}"
        )
        print(
            f"Model σ: {model_std:.{print_decimals}f} {unit}"
        )

    if xlabel is None:
        xlabel = (
            f"{parameter_name} [{unit}]"
        )

    if title is None:
        title = (
            f"Pooled {parameter_name} distribution"
        )

    figure, axes = plot_pooled_histogram_with_model(
        pooled_values=pooled_values,
        mean=model_mean,
        std=model_std,
        distribution=distribution,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        bins=bins,
        n_sigmas=n_sigmas,
        x_limits=x_limits,
        save_path=save_path,
        dpi=dpi,
        n_model_points=n_model_points,
    )

    if show:
        plt.show()

    return ParameterCalibrationResult(
        maps=maps,
        subject_values=subject_values,
        pooled_values=pooled_values,
        median=median,
        iqr=iqr,
        model_mean=model_mean,
        model_std=model_std,
        model_parameter_space=model_parameter_space,
        figure=figure,
        axes=axes,
    )