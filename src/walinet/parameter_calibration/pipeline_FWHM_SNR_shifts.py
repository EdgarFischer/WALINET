from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np

from walinet.parameter_calibration.compute_statistics import (
    extract_valid_voxels,
)
from walinet.parameter_calibration.load_data import (
    load_subject_maps,
)
from walinet.parameter_calibration.plot_statistics import (
    compare_positive_models,
    compare_symmetric_models,
)


DistributionModel = Literal[
    "positive_mixture",
    "symmetric_mixture",
]


@dataclass(frozen=True)
class ParameterCalibrationResult:
    """
    Result of the common mixture-model calibration.

    Positive parameters
    -------------------
    ``distribution == "positive_mixture"`` uses:

        0.5 * ZeroTruncatedNormal(normal_mu, normal_sigma)
        +
        0.5 * LogNormal(log_mu, log_sigma)

    Signed parameters
    -----------------
    ``distribution == "symmetric_mixture"`` uses:

        0.5 * Normal(center, normal_sigma)
        +
        0.25 * positive LogNormal(log_mu, log_sigma)
        +
        0.25 * negative LogNormal(log_mu, log_sigma)

    For the symmetric model, the lognormal distribution is calibrated
    on ``abs(values - center)``, where ``center`` is the pooled median.
    """

    maps: np.ndarray
    subject_values: dict[str, np.ndarray]
    pooled_values: np.ndarray

    distribution: DistributionModel

    median: float
    iqr: float

    normal_mu: float
    normal_sigma: float
    normal_sigma_factor: float

    log_mu: float
    log_sigma: float
    robust_log_sigma: float
    lognormal_sigma_factor: float

    center: float | None

    normal_weight: float
    lognormal_weight: float
    positive_tail_weight: float | None
    negative_tail_weight: float | None

    plot_xmin: float
    plot_xmax: float

    figure: plt.Figure
    axes: plt.Axes


def _filter_subject_values(
    subject_values: dict[str, np.ndarray],
    *,
    positive_only: bool,
) -> dict[str, np.ndarray]:
    """Apply the same validity criterion to every subject array."""
    filtered: dict[str, np.ndarray] = {}

    for subject, values in subject_values.items():
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        valid = np.isfinite(
            values
        )

        if positive_only:
            valid &= values > 0

        filtered[subject] = values[
            valid
        ]

    return filtered


def calibrate_parameter_from_maps(
    *,
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    quality_mask: np.ndarray,
    parameter_name: str,
    unit: str,
    extension: str = ".nii.gz",
    distribution: DistributionModel,
    xlabel: str | None = None,
    ylabel: str = "Probability density",
    title: str | None = None,
    bins: int | str = 50,
    normal_sigma_factor: float = 1.0,
    lognormal_sigma_factor: float = 1.0,
    plot_percentile: float = 99.5,
    x_limits: tuple[float, float] | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1200,
    show: bool = True,
) -> ParameterCalibrationResult:
    """
    Load parameter maps and calibrate the common simulation mixture.

    Supported models
    ----------------
    ``distribution="positive_mixture"``
        For strictly positive parameters such as FWHM and SNR:

            0.5 * zero-truncated normal
            +
            0.5 * lognormal

        Calibration:

            normal_mu = median(values)

            robust_normal_sigma = IQR(values) / 1.349

            normal_sigma = (
                normal_sigma_factor
                * robust_normal_sigma
            )

            log_mu = median(log(values))

            robust_log_sigma = (
                IQR(log(values)) / 1.349
            )

            log_sigma = (
                lognormal_sigma_factor
                * robust_log_sigma
            )

    ``distribution="symmetric_mixture"``
        For signed parameters such as frequency shift and phase:

            0.5 * central normal
            +
            0.25 * positive lognormal tail
            +
            0.25 * negative lognormal tail

        Calibration:

            center = median(values)

            robust_normal_sigma = IQR(values) / 1.349

            normal_sigma = (
                normal_sigma_factor
                * robust_normal_sigma
            )

            deviations = abs(values - center)

            log_mu = median(log(deviations))

            robust_log_sigma = (
                IQR(log(deviations)) / 1.349
            )

            log_sigma = (
                lognormal_sigma_factor
                * robust_log_sigma
            )

    The mixture weights are fixed globally and are not fitted separately
    for individual parameters.
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
        "positive_mixture",
        "symmetric_mixture",
    }

    if distribution not in valid_distributions:
        raise ValueError(
            f"Unsupported distribution: {distribution!r}. "
            f"Supported values are: {sorted(valid_distributions)}."
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

    positive_only = (
        distribution == "positive_mixture"
    )

    subject_values = _filter_subject_values(
        subject_values,
        positive_only=positive_only,
    )

    pooled_values = np.asarray(
        pooled_values,
        dtype=np.float64,
    )

    pooled_valid = np.isfinite(
        pooled_values
    )

    if positive_only:
        pooled_valid &= pooled_values > 0

    number_removed = int(
        pooled_values.size
        - np.count_nonzero(
            pooled_valid
        )
    )

    pooled_values = pooled_values[
        pooled_valid
    ]

    if pooled_values.size == 0:
        if positive_only:
            raise ValueError(
                "No strictly positive finite pooled values are "
                "available for positive-mixture calibration."
            )

        raise ValueError(
            "No finite pooled values are available for "
            "symmetric-mixture calibration."
        )

    if number_removed > 0:
        criterion = (
            "non-positive or non-finite"
            if positive_only
            else "non-finite"
        )

        print(
            f"Ignored {number_removed} {criterion} values."
        )

    if xlabel is None:
        xlabel = (
            f"{parameter_name} [{unit}]"
        )

    if title is None:
        title = (
            f"Pooled {parameter_name} distribution"
        )

    print(
        f"\n{parameter_name} [{unit}]"
    )
    print(
        "=" * (
            len(parameter_name)
            + len(unit)
            + 3
        )
    )

    if distribution == "positive_mixture":
        statistics = compare_positive_models(
            pooled_values,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            bins=bins,
            truncated_normal_sigma_factor=normal_sigma_factor,
            lognormal_sigma_factor=lognormal_sigma_factor,
            plot_percentile=plot_percentile,
            x_limits=x_limits,
            save_path=save_path,
            dpi=dpi,
            n_model_points=n_model_points,
            show=show,
        )

        center: float | None = None
        positive_tail_weight: float | None = None
        negative_tail_weight: float | None = None

    else:
        statistics = compare_symmetric_models(
            pooled_values,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            bins=bins,
            normal_sigma_factor=normal_sigma_factor,
            lognormal_sigma_factor=lognormal_sigma_factor,
            plot_percentile=plot_percentile,
            x_limits=x_limits,
            save_path=save_path,
            dpi=dpi,
            n_model_points=n_model_points,
            show=show,
        )

        center = float(
            statistics["center"]
        )
        positive_tail_weight = float(
            statistics["positive_tail_weight"]
        )
        negative_tail_weight = float(
            statistics["negative_tail_weight"]
        )

    figure = statistics[
        "figure"
    ]
    axes = statistics[
        "axes"
    ]

    if not isinstance(
        figure,
        plt.Figure,
    ):
        raise TypeError(
            "The plotting function returned an invalid figure."
        )

    return ParameterCalibrationResult(
        maps=maps,
        subject_values=subject_values,
        pooled_values=pooled_values,
        distribution=distribution,
        median=float(
            statistics["median"]
        ),
        iqr=float(
            statistics["iqr"]
        ),
        normal_mu=float(
            statistics["normal_mu"]
        ),
        normal_sigma=float(
            statistics["normal_sigma"]
        ),
        normal_sigma_factor=float(
            statistics["normal_sigma_factor"]
        ),
        log_mu=float(
            statistics["log_mu"]
        ),
        log_sigma=float(
            statistics["log_sigma"]
        ),
        robust_log_sigma=float(
            statistics["robust_log_sigma"]
        ),
        lognormal_sigma_factor=float(
            statistics["log_sigma_factor"]
        ),
        center=center,
        normal_weight=float(
            statistics["normal_weight"]
        ),
        lognormal_weight=float(
            statistics["lognormal_weight"]
        ),
        positive_tail_weight=positive_tail_weight,
        negative_tail_weight=negative_tail_weight,
        plot_xmin=float(
            statistics["plot_xmin"]
        ),
        plot_xmax=float(
            statistics["plot_xmax"]
        ),
        figure=figure,
        axes=axes,
    )