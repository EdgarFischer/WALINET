from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np

from walinet.parameter_calibration.load_data import (
    load_subject_maps,
)
from walinet.parameter_calibration.plot_statistics import (
    compare_positive_models,
)


@dataclass(frozen=True)
class MetaboliteReferenceData:
    """
    Data shared by all metabolite-ratio calibrations.
    """

    brain_mask: np.ndarray
    metabolites: np.ndarray


@dataclass(frozen=True)
class MetaboliteRatioCalibrationResult:
    """
    Complete calibration result for the positive 50/50 mixture model.

    The zero-truncated normal component is calibrated as:

        normal_mu = median(values)
        robust_normal_sigma = IQR(values) / 1.349
        normal_sigma = (
            truncated_normal_sigma_factor
            * robust_normal_sigma
        )

    The lognormal component is calibrated as:

        log_mu = median(log(values))
        robust_log_sigma = IQR(log(values)) / 1.349
        log_sigma = (
            lognormal_sigma_factor
            * robust_log_sigma
        )

    The final sampling model uses fixed global weights:

        0.5 * ZeroTruncatedNormal
        +
        0.5 * LogNormal
    """

    coefficient_maps: np.ndarray
    ratio_maps: np.ndarray
    pooled_values: np.ndarray

    median: float
    iqr: float

    normal_mu: float
    robust_normal_sigma: float
    truncated_normal_sigma_factor: float
    normal_sigma: float

    log_mu: float
    robust_log_sigma: float
    lognormal_sigma_factor: float
    log_sigma: float
    lognormal_median: float

    normal_weight: float
    lognormal_weight: float

    plot_xmin: float
    plot_xmax: float

    figure: Figure
    axes: Axes


def load_metabolite_reference_data(
    *,
    base_paths: Sequence[str | Path],
    mask_relative_path: str | Path = "maps/mask",
    fit_relative_path: str | Path = "maps/SpecMap_LCMFit",
    baseline_relative_path: str | Path = "maps/SpecMap_LCMBaseline",
    extension: str = ".nii.gz",
    spectral_axis: int = -2,
) -> MetaboliteReferenceData:
    """
    Load the brain mask and reconstruct the fitted metabolite spectra:

        metabolites = SpecMap_Fit - SpecMap_Baseline

    These arrays can be reused for all metabolite coefficient maps.
    """
    brain_mask = load_subject_maps(
        base_paths=base_paths,
        relative_path=mask_relative_path,
        extension=extension,
    )

    spec_map_fit = load_subject_maps(
        base_paths=base_paths,
        relative_path=fit_relative_path,
        extension=extension,
    )

    spec_map_baseline = load_subject_maps(
        base_paths=base_paths,
        relative_path=baseline_relative_path,
        extension=extension,
    )

    if spec_map_fit.shape != spec_map_baseline.shape:
        raise ValueError(
            "SpecMap_Fit and SpecMap_Baseline must have the same "
            "shape.\n"
            f"  SpecMap_Fit:      {spec_map_fit.shape}\n"
            f"  SpecMap_Baseline: {spec_map_baseline.shape}"
        )

    metabolites = (
        spec_map_fit
        - spec_map_baseline
    )

    metabolite_maximum = np.max(
        np.abs(metabolites),
        axis=spectral_axis,
    )

    if brain_mask.shape != metabolite_maximum.shape:
        raise ValueError(
            "brain_mask does not match the spatial and subject "
            "dimensions of the metabolite spectra.\n"
            f"  brain_mask:         {brain_mask.shape}\n"
            f"  metabolite maximum: {metabolite_maximum.shape}\n"
            f"  metabolites:        {metabolites.shape}\n"
            f"  spectral_axis:      {spectral_axis}"
        )

    print("Loaded metabolite reference data:")
    print(
        f"  brain_mask: {brain_mask.shape}"
    )
    print(
        f"  metabolites: {metabolites.shape}"
    )

    return MetaboliteReferenceData(
        brain_mask=brain_mask,
        metabolites=metabolites,
    )


def calculate_metabolite_ratio(
    coefficient_maps: np.ndarray,
    metabolites: np.ndarray,
    brain_mask: np.ndarray,
    *,
    spectral_axis: int = -2,
) -> np.ndarray:
    """
    Calculate voxel-wise:

        coefficient / max(abs(metabolite spectrum))

    ``brain_mask`` may already contain additional quality criteria,
    such as a metabolite-specific CRLB threshold.
    """
    coefficient_maps = np.asarray(
        coefficient_maps,
    )

    metabolites = np.asarray(
        metabolites,
    )

    brain_mask = np.asarray(
        brain_mask,
    )

    metabolite_maximum = np.max(
        np.abs(metabolites),
        axis=spectral_axis,
    )

    if coefficient_maps.shape != metabolite_maximum.shape:
        raise ValueError(
            "The coefficient-map shape does not match the metabolite "
            "maximum shape.\n"
            f"  coefficient_maps:   {coefficient_maps.shape}\n"
            f"  metabolite maximum: {metabolite_maximum.shape}"
        )

    if brain_mask.shape != metabolite_maximum.shape:
        raise ValueError(
            "The brain-mask shape does not match the metabolite "
            "maximum shape.\n"
            f"  brain_mask:         {brain_mask.shape}\n"
            f"  metabolite maximum: {metabolite_maximum.shape}"
        )

    valid = (
        brain_mask.astype(bool)
        & np.isfinite(coefficient_maps)
        & np.isfinite(metabolite_maximum)
        & (coefficient_maps > 0)
        & (metabolite_maximum > 0)
    )

    ratio_maps = np.full(
        metabolite_maximum.shape,
        np.nan,
        dtype=np.float32,
    )

    ratio_maps[valid] = (
        coefficient_maps[valid]
        / metabolite_maximum[valid]
    )

    return ratio_maps


def calibrate_metabolite_ratio_from_map(
    *,
    reference_data: MetaboliteReferenceData,
    base_paths: Sequence[str | Path],
    relative_path: str | Path,
    metabolite_name: str,
    extension: str = ".nii.gz",
    crlb_threshold: float | None = 30.0,
    spectral_axis: int = -2,
    bins: int | str = 50,
    plot_percentile: float = 99.5,
    truncated_normal_sigma_factor: float = 1.0,
    lognormal_sigma_factor: float = 1.0,
    title: str | None = None,
    xlabel: str | None = None,
    save_path: str | Path | None = None,
    dpi: int = 300,
    show: bool = True,
) -> MetaboliteRatioCalibrationResult:
    """
    Load one metabolite amplitude map and run the complete
    ratio-calibration pipeline.

    The corresponding metabolite-specific CRLB map is inferred
    automatically by replacing:

        "_amp_map" -> "_sd_map"

    in ``relative_path``.

    When ``crlb_threshold`` is not None, only voxels satisfying

        CRLB <= crlb_threshold

    are used, in addition to the brain mask.

    The positive ratio distribution is represented by the common
    fixed-weight mixture model:

        0.5 * ZeroTruncatedNormal
        +
        0.5 * LogNormal

    Both components are calibrated robustly from the same in-vivo
    values. For the zero-truncated normal component:

        normal_mu = median(ratio)
        robust_normal_sigma = IQR(ratio) / 1.349
        normal_sigma = (
            truncated_normal_sigma_factor
            * robust_normal_sigma
        )

    For the lognormal component:

        log_mu = median(log(ratio))
        robust_log_sigma = IQR(log(ratio)) / 1.349
        log_sigma = (
            lognormal_sigma_factor
            * robust_log_sigma
        )

    Set both sigma factors to 1.0 for the final directly calibrated
    model without deliberate broadening.
    """
    if not metabolite_name.strip():
        raise ValueError(
            "metabolite_name must not be empty."
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
            "lognormal_sigma_factor must be finite and greater "
            "than zero."
        )

    relative_path = Path(
        relative_path
    )

    # ---------------------------------------------------------
    # Load metabolite coefficient maps
    # ---------------------------------------------------------
    coefficient_maps = load_subject_maps(
        base_paths=base_paths,
        relative_path=relative_path,
        extension=extension,
    )

    combined_mask = np.asarray(
        reference_data.brain_mask,
        dtype=bool,
    ).copy()

    if combined_mask.shape != coefficient_maps.shape:
        raise ValueError(
            "The brain mask and coefficient maps must have the "
            "same shape.\n"
            f"  brain mask:       {combined_mask.shape}\n"
            f"  coefficient maps: {coefficient_maps.shape}"
        )

    # ---------------------------------------------------------
    # Load and apply metabolite-specific CRLB map
    # ---------------------------------------------------------
    if crlb_threshold is not None:
        crlb_threshold = float(
            crlb_threshold
        )

        if (
            not np.isfinite(crlb_threshold)
            or crlb_threshold <= 0
        ):
            raise ValueError(
                "crlb_threshold must be finite and greater than zero, "
                "or None to disable CRLB filtering."
            )

        if "_amp_map" not in relative_path.name:
            raise ValueError(
                "Cannot infer the CRLB path because relative_path "
                "does not contain '_amp_map'.\n"
                f"  relative_path: {relative_path}"
            )

        crlb_relative_path = relative_path.with_name(
            relative_path.name.replace(
                "_amp_map",
                "_sd_map",
                1,
            )
        )

        crlb_maps = load_subject_maps(
            base_paths=base_paths,
            relative_path=crlb_relative_path,
            extension=extension,
        )

        if crlb_maps.shape != coefficient_maps.shape:
            raise ValueError(
                "The CRLB maps and coefficient maps must have the "
                "same shape.\n"
                f"  coefficient maps: {coefficient_maps.shape}\n"
                f"  CRLB maps:        {crlb_maps.shape}\n"
                f"  CRLB path:        {crlb_relative_path}"
            )

        crlb_mask = (
            np.isfinite(crlb_maps)
            & (crlb_maps >= 0)
            & (crlb_maps <= crlb_threshold)
        )

        brain_voxel_count = int(
            np.count_nonzero(
                combined_mask
            )
        )

        combined_mask &= crlb_mask

        retained_voxel_count = int(
            np.count_nonzero(
                combined_mask
            )
        )

        retained_fraction = (
            retained_voxel_count
            / brain_voxel_count
            if brain_voxel_count > 0
            else 0.0
        )

        print(
            f"{metabolite_name} CRLB filtering:"
        )
        print(
            f"  CRLB map:       {crlb_relative_path}"
        )
        print(
            f"  threshold:      <= {crlb_threshold:g} %"
        )
        print(
            f"  brain voxels:   {brain_voxel_count}"
        )
        print(
            f"  retained:       {retained_voxel_count} "
            f"({100.0 * retained_fraction:.1f} %)"
        )

    else:
        print(
            f"{metabolite_name}: CRLB filtering disabled."
        )

    # ---------------------------------------------------------
    # Calculate coefficient / maximum metabolite signal
    # ---------------------------------------------------------
    ratio_maps = calculate_metabolite_ratio(
        coefficient_maps=coefficient_maps,
        metabolites=reference_data.metabolites,
        brain_mask=combined_mask,
        spectral_axis=spectral_axis,
    )

    pooled_values = ratio_maps[
        np.isfinite(ratio_maps)
        & (ratio_maps > 0)
    ].astype(
        np.float64,
        copy=False,
    )

    if pooled_values.size == 0:
        raise ValueError(
            f"No valid positive ratios were found for "
            f"{metabolite_name!r}."
        )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    if title is None:
        title = (
            f"{metabolite_name} coefficient ratio"
        )

    if xlabel is None:
        xlabel = (
            f"{metabolite_name} coefficient / "
            "max|Metabolites|"
        )

    # ---------------------------------------------------------
    # Estimate and plot distributions
    # ---------------------------------------------------------
    statistics = compare_positive_models(
        pooled_values,
        title=title,
        xlabel=xlabel,
        bins=bins,
        plot_percentile=plot_percentile,
        truncated_normal_sigma_factor=(
            truncated_normal_sigma_factor
        ),
        lognormal_sigma_factor=(
            lognormal_sigma_factor
        ),
        save_path=save_path,
        dpi=dpi,
        show=show,
    )

    normal_sigma_factor = float(
        statistics["normal_sigma_factor"]
    )

    normal_sigma = float(
        statistics["normal_sigma"]
    )

    robust_normal_sigma = (
        normal_sigma
        / normal_sigma_factor
    )

    result = MetaboliteRatioCalibrationResult(
        coefficient_maps=coefficient_maps,
        ratio_maps=ratio_maps,
        pooled_values=pooled_values,
        median=float(
            statistics["median"]
        ),
        iqr=float(
            statistics["iqr"]
        ),
        normal_mu=float(
            statistics["normal_mu"]
        ),
        robust_normal_sigma=float(
            robust_normal_sigma
        ),
        truncated_normal_sigma_factor=(
            normal_sigma_factor
        ),
        normal_sigma=normal_sigma,
        log_mu=float(
            statistics["log_mu"]
        ),
        robust_log_sigma=float(
            statistics["robust_log_sigma"]
        ),
        lognormal_sigma_factor=float(
            statistics["log_sigma_factor"]
        ),
        log_sigma=float(
            statistics["log_sigma"]
        ),
        lognormal_median=float(
            np.exp(
                statistics["log_mu"]
            )
        ),
        normal_weight=float(
            statistics["normal_weight"]
        ),
        lognormal_weight=float(
            statistics["lognormal_weight"]
        ),
        plot_xmin=float(
            statistics["plot_xmin"]
        ),
        plot_xmax=float(
            statistics["plot_xmax"]
        ),
        figure=statistics["figure"],
        axes=statistics["axes"],
    )

    print(
        f"\nFinal {metabolite_name} positive-mixture parameters:"
    )

    print(
        "  Zero-truncated normal component:"
    )
    print(
        f"    mu                    = {result.normal_mu:.6f}"
    )
    print(
        f"    robust_sigma          = "
        f"{result.robust_normal_sigma:.6f}"
    )
    print(
        f"    sigma_factor          = "
        f"{result.truncated_normal_sigma_factor:.6f}"
    )
    print(
        f"    final_sigma           = {result.normal_sigma:.6f}"
    )

    print(
        "  Lognormal component:"
    )
    print(
        f"    log_mu                = {result.log_mu:.6f}"
    )
    print(
        f"    robust_log_sigma      = "
        f"{result.robust_log_sigma:.6f}"
    )
    print(
        f"    sigma_factor          = "
        f"{result.lognormal_sigma_factor:.6f}"
    )
    print(
        f"    final_log_sigma       = {result.log_sigma:.6f}"
    )
    print(
        f"    median                = "
        f"{result.lognormal_median:.6f}"
    )

    print(
        "  Mixture:"
    )
    print(
        f"    normal_weight         = {result.normal_weight:.6f}"
    )
    print(
        f"    lognormal_weight      = {result.lognormal_weight:.6f}"
    )
    print(
        f"    voxels                = {result.pooled_values.size}"
    )

    return result