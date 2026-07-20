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
]


@dataclass(frozen=True)
class ParameterCalibrationResult:
    """
    Results produced by the parameter-calibration pipeline.
    """

    maps: np.ndarray
    subject_values: dict[str, np.ndarray]
    pooled_values: np.ndarray

    median: float
    iqr: float

    model_mean: float
    model_std: float

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

    The model parameters used for the histogram overlay are:

        model_mean = pooled median
        model_std = std_iqr_factor * pooled IQR

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

        Example:
            "FWHM"

    unit:
        Unit shown in the printed output and axis label.

        Example:
            "ppm"

    extension:
        File extension passed to ``load_subject_maps``.

    distribution:
        Probability model overlaid on the histogram.

        Supported values:
            "normal"
            "truncated_normal"

    std_iqr_factor:
        Factor used to convert the pooled IQR into the model standard
        deviation:

            model_std = std_iqr_factor * IQR

        For example, ``std_iqr_factor=2.0`` gives:

            model_std = 2 * IQR

    xlabel:
        Optional custom x-axis label.

        When omitted, the label is generated automatically as:

            "<parameter_name> [<unit>]"

    ylabel:
        Label of the y-axis.

    title:
        Optional plot title.

        When omitted, the title is generated automatically as:

            "Pooled <parameter_name> distribution"

    bins:
        Number of histogram bins or NumPy binning strategy.

    n_sigmas:
        Number of model standard deviations shown around the model
        mean. Ignored when ``x_limits`` is provided.

    x_limits:
        Optional explicit x-axis limits.

    save_path:
        Optional path under which the plot is saved.

    dpi:
        Resolution used when saving raster images.

    n_model_points:
        Number of points used to draw the model density.

    print_decimals:
        Number of decimal places used when printing median and IQR.

    show:
        Whether to call ``plt.show()`` after creating the plot.

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

    if print_decimals < 0:
        raise ValueError(
            "print_decimals must be greater than or equal to zero."
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

    median, iqr = calculate_pooled_median_iqr(
        pooled_values
    )

    model_mean = median
    model_std = std_iqr_factor * iqr

    if not np.isfinite(model_std) or model_std <= 0:
        raise ValueError(
            "The calculated model standard deviation must be "
            "greater than zero.\n"
            f"  IQR:            {iqr}\n"
            f"  std_iqr_factor: {std_iqr_factor}\n"
            f"  model_std:      {model_std}"
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
        figure=figure,
        axes=axes,
    )