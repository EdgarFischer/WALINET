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


from pathlib import Path
from collections.abc import Mapping, Sequence
import re

import numpy as np

from walinet.parameter_calibration.metab_calibration import (
    MetaboliteRatioCalibrationResult,
    compare_positive_models,
)


def calibrate_metabolite_ratio_from_r_maps(
    *,
    r_calibration: Mapping,
    metabolite_name: str,
    subject_ids: Sequence[str] | None = None,
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
    Calibrate the positive r_i distribution of one metabolite.

    Subjects without the requested metabolite map are skipped.
    Different spatial map shapes are supported.

    Statistics are calculated jointly from all positive finite
    voxels of all subjects for which the map is available.
    """

    def name_key(name: str) -> str:
        name = re.sub(
            r"^\d+[_-]+",
            "",
            str(name),
        )

        return re.sub(
            r"[^a-z0-9]",
            "",
            name.lower(),
        )

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------
    if not isinstance(r_calibration, Mapping) or not r_calibration:
        raise ValueError(
            "r_calibration must be a non-empty mapping."
        )

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
            "truncated_normal_sigma_factor must be finite "
            "and greater than zero."
        )

    lognormal_sigma_factor = float(
        lognormal_sigma_factor
    )

    if (
        not np.isfinite(lognormal_sigma_factor)
        or lognormal_sigma_factor <= 0
    ):
        raise ValueError(
            "lognormal_sigma_factor must be finite "
            "and greater than zero."
        )

    # ---------------------------------------------------------
    # Select subjects
    # ---------------------------------------------------------
    if subject_ids is None:
        selected_subject_ids = list(
            r_calibration.keys()
        )

    else:
        selected_subject_ids = list(
            subject_ids
        )

        if not selected_subject_ids:
            raise ValueError(
                "subject_ids must not be empty."
            )

        unknown_subject_ids = [
            subject_id
            for subject_id in selected_subject_ids
            if subject_id not in r_calibration
        ]

        if unknown_subject_ids:
            raise KeyError(
                "Unknown subject IDs:\n  "
                + "\n  ".join(
                    unknown_subject_ids
                )
            )

    target_key = name_key(
        metabolite_name
    )

    coefficient_maps_by_subject = []
    ratio_maps_by_subject = []
    pooled_values_by_subject = []

    used_subject_ids = []
    skipped_subject_ids = []

    canonical_basis_name = None

    # ---------------------------------------------------------
    # Extract available subject maps
    # ---------------------------------------------------------
    for subject_id in selected_subject_ids:
        subject = r_calibration[
            subject_id
        ]

        required_keys = {
            "r_maps",
            "coefficients",
            "brain_mask",
            "basis_names",
        }

        missing_keys = sorted(
            required_keys.difference(
                subject
            )
        )

        if missing_keys:
            raise KeyError(
                f"{subject_id}: missing entries in "
                f"r_calibration: {missing_keys}"
            )

        basis_names = list(
            subject["basis_names"]
        )

        matching_indices = [
            index
            for index, basis_name in enumerate(
                basis_names
            )
            if name_key(
                basis_name
            ) == target_key
        ]

        if not matching_indices:
            raise ValueError(
                f"{subject_id}: metabolite "
                f"{metabolite_name!r} was not found "
                "in basis_names."
            )

        if len(matching_indices) > 1:
            matching_names = [
                basis_names[index]
                for index in matching_indices
            ]

            raise ValueError(
                f"{subject_id}: metabolite name "
                f"{metabolite_name!r} is ambiguous: "
                f"{matching_names}"
            )

        basis_index = matching_indices[
            0
        ]

        basis_name = basis_names[
            basis_index
        ]

        if canonical_basis_name is None:
            canonical_basis_name = basis_name

        # -----------------------------------------------------
        # Skip subject if no coefficient map was loaded
        # -----------------------------------------------------
        matched_basis_names = subject.get(
            "matched_basis_names"
        )

        if matched_basis_names is not None:
            matched_keys = {
                name_key(name)
                for name in matched_basis_names
            }

            if target_key not in matched_keys:
                skipped_subject_ids.append(
                    subject_id
                )
                continue

        r_maps = np.asarray(
            subject["r_maps"],
            dtype=np.float32,
        )

        coefficients = np.asarray(
            subject["coefficients"],
            dtype=np.float32,
        )

        brain_mask = np.asarray(
            subject["brain_mask"],
            dtype=bool,
        )

        expected_shape = (
            *brain_mask.shape,
            len(basis_names),
        )

        if r_maps.shape != expected_shape:
            raise ValueError(
                f"{subject_id}: r_maps has shape "
                f"{r_maps.shape}, expected "
                f"{expected_shape}."
            )

        if coefficients.shape != expected_shape:
            raise ValueError(
                f"{subject_id}: coefficients has shape "
                f"{coefficients.shape}, expected "
                f"{expected_shape}."
            )

        ratio_map = r_maps[
            ...,
            basis_index,
        ].copy()

        coefficient_map = coefficients[
            ...,
            basis_index,
        ].copy()

        ratio_map[
            ~brain_mask
        ] = np.nan

        coefficient_map[
            ~brain_mask
        ] = np.nan

        valid_values = ratio_map[
            np.isfinite(ratio_map)
            & (ratio_map > 0)
        ].astype(
            np.float64,
            copy=False,
        )

        coefficient_maps_by_subject.append(
            coefficient_map
        )

        ratio_maps_by_subject.append(
            ratio_map
        )

        if valid_values.size > 0:
            pooled_values_by_subject.append(
                valid_values
            )

        used_subject_ids.append(
            subject_id
        )

    # ---------------------------------------------------------
    # Validate available data
    # ---------------------------------------------------------
    if not used_subject_ids:
        raise ValueError(
            f"The map for {metabolite_name!r} "
            "was not loaded and matched for any "
            "selected subject."
        )

    if not pooled_values_by_subject:
        raise ValueError(
            f"No valid positive ratios were found for "
            f"{canonical_basis_name!r}."
        )

    pooled_values = np.concatenate(
        pooled_values_by_subject,
        axis=0,
    )

    # Different spatial shapes cannot be stacked into a regular
    # numeric array. Keep one map object per subject instead.
    coefficient_maps = np.empty(
        len(coefficient_maps_by_subject),
        dtype=object,
    )

    coefficient_maps[:] = (
        coefficient_maps_by_subject
    )

    ratio_maps = np.empty(
        len(ratio_maps_by_subject),
        dtype=object,
    )

    ratio_maps[:] = ratio_maps_by_subject

    # ---------------------------------------------------------
    # Report skipped subjects
    # ---------------------------------------------------------
    if skipped_subject_ids:
        print(
            f"\n{canonical_basis_name}: skipped "
            f"{len(skipped_subject_ids)} subject(s) "
            "without a matched map:"
        )

        for subject_id in skipped_subject_ids:
            print(
                f"  {subject_id}"
            )

    # ---------------------------------------------------------
    # Labels
    # ---------------------------------------------------------
    if title is None:
        title = (
            f"{canonical_basis_name} coefficient ratio"
        )

    if xlabel is None:
        xlabel = (
            f"{canonical_basis_name} coefficient / "
            "max|Metabolites|"
        )

    # ---------------------------------------------------------
    # Estimate and plot distribution
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

    # ---------------------------------------------------------
    # Construct result
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Parameter output
    # ---------------------------------------------------------
    print(
        f"\nFinal {canonical_basis_name} "
        "positive-mixture parameters:"
    )

    print(
        "  Zero-truncated normal component:"
    )
    print(
        f"    mu                    = "
        f"{result.normal_mu:.6f}"
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
        f"    final_sigma           = "
        f"{result.normal_sigma:.6f}"
    )

    print(
        "  Lognormal component:"
    )
    print(
        f"    log_mu                = "
        f"{result.log_mu:.6f}"
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
        f"    final_log_sigma       = "
        f"{result.log_sigma:.6f}"
    )
    print(
        f"    median                = "
        f"{result.lognormal_median:.6f}"
    )

    print(
        "  Mixture:"
    )
    print(
        f"    normal_weight         = "
        f"{result.normal_weight:.6f}"
    )
    print(
        f"    lognormal_weight      = "
        f"{result.lognormal_weight:.6f}"
    )
    print(
        f"    subjects used         = "
        f"{len(used_subject_ids)}"
    )
    print(
        f"    subjects skipped      = "
        f"{len(skipped_subject_ids)}"
    )
    print(
        f"    voxels                = "
        f"{result.pooled_values.size}"
    )

    print(
        "    subject shapes:"
    )

    for subject_id, ratio_map in zip(
        used_subject_ids,
        ratio_maps_by_subject,
    ):
        print(
            f"      {subject_id}: "
            f"{ratio_map.shape}"
        )

    return result

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


def plot_fwhm_vs_metabolite_slices(
    calibration_maps,
    *,
    metabolite_name="NAA",
    slices_per_figure=6,
    slice_axis=2,
    percentile_range=(1.0, 99.0),
    save_dir=None,
    dpi=200,
    show=True,
):
    """
    Plot all slices of one metabolite amplitude map next to the
    corresponding FWHM map for every subject.

    For each slice:

        left:  metabolite amplitude map
        right: FWHM map

    Parameters
    ----------
    calibration_maps:
        Output of ``load_calibration_maps``.

    metabolite_name:
        Metabolite to compare with FWHM, for example ``"NAA"``.

    slices_per_figure:
        Number of slices shown in one figure.

    slice_axis:
        Spatial axis along which slices are extracted.
        For maps with shape (X, Y, Z), use 2.

    percentile_range:
        Robust lower and upper percentiles used for the color scales.

    save_dir:
        Optional directory for saving PNG figures.

    Returns
    -------
    figures:
        Dictionary mapping each subject ID to a list of generated figures.
    """

    def name_key(name):
        name = re.sub(
            r"^\d+[_-]+",
            "",
            str(name),
        )

        return re.sub(
            r"[^a-z0-9]",
            "",
            name.lower(),
        )

    def robust_limits(values, mask):
        valid_values = np.asarray(values)[
            mask
            & np.isfinite(values)
        ]

        if valid_values.size == 0:
            return 0.0, 1.0

        lower, upper = np.percentile(
            valid_values,
            percentile_range,
        )

        lower = float(lower)
        upper = float(upper)

        if not np.isfinite(lower):
            lower = 0.0

        if (
            not np.isfinite(upper)
            or upper <= lower
        ):
            upper = lower + 1.0

        return lower, upper

    if not calibration_maps:
        raise ValueError(
            "calibration_maps must not be empty."
        )

    if slices_per_figure <= 0:
        raise ValueError(
            "slices_per_figure must be greater than zero."
        )

    if save_dir is not None:
        save_dir = Path(
            save_dir
        )

        save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    target_key = name_key(
        metabolite_name
    )

    figures = {}

    for subject_id, subject_data in calibration_maps.items():

        metabolite_maps = subject_data[
            "metabolites"
        ]

        matching_names = [
            name
            for name in metabolite_maps
            if name_key(name) == target_key
        ]

        if not matching_names:
            available_names = ", ".join(
                metabolite_maps.keys()
            )

            print(
                f"Skipping {subject_id}: "
                f"{metabolite_name!r} not found.\n"
                f"  Available: {available_names}"
            )

            continue

        if len(matching_names) > 1:
            raise ValueError(
                f"{subject_id}: multiple maps match "
                f"{metabolite_name!r}: {matching_names}"
            )

        matched_name = matching_names[
            0
        ]

        metabolite_map = np.asarray(
            metabolite_maps[
                matched_name
            ],
            dtype=np.float32,
        )

        fwhm_map = np.asarray(
            subject_data[
                "fwhm"
            ],
            dtype=np.float32,
        )

        if metabolite_map.shape != fwhm_map.shape:
            raise ValueError(
                f"{subject_id}: shape mismatch:\n"
                f"  {matched_name}: {metabolite_map.shape}\n"
                f"  FWHM:       {fwhm_map.shape}"
            )

        if slice_axis not in {
            0,
            1,
            2,
        }:
            raise ValueError(
                "slice_axis must be 0, 1, or 2."
            )

        n_slices = metabolite_map.shape[
            slice_axis
        ]

        # Mask background voxels. This is derived from the available
        # LCModel maps rather than from an external anatomical mask.
        valid_mask = (
            np.isfinite(
                metabolite_map
            )
            & np.isfinite(
                fwhm_map
            )
            & (
                (
                    metabolite_map != 0
                )
                | (
                    fwhm_map != 0
                )
            )
        )

        metabolite_vmin, metabolite_vmax = robust_limits(
            metabolite_map,
            valid_mask,
        )

        fwhm_vmin, fwhm_vmax = robust_limits(
            fwhm_map,
            valid_mask,
        )

        subject_figures = []

        for first_slice in range(
            0,
            n_slices,
            slices_per_figure,
        ):
            last_slice = min(
                first_slice + slices_per_figure,
                n_slices,
            )

            slice_indices = list(
                range(
                    first_slice,
                    last_slice,
                )
            )

            n_rows = len(
                slice_indices
            )

            fig, axes = plt.subplots(
                nrows=n_rows,
                ncols=2,
                figsize=(
                    9,
                    3.5 * n_rows,
                ),
                squeeze=False,
            )

            metabolite_image = None
            fwhm_image = None

            for row, slice_index in enumerate(
                slice_indices
            ):
                metabolite_slice = np.take(
                    metabolite_map,
                    slice_index,
                    axis=slice_axis,
                )

                fwhm_slice = np.take(
                    fwhm_map,
                    slice_index,
                    axis=slice_axis,
                )

                mask_slice = np.take(
                    valid_mask,
                    slice_index,
                    axis=slice_axis,
                )

                metabolite_slice = np.where(
                    mask_slice,
                    metabolite_slice,
                    np.nan,
                )

                fwhm_slice = np.where(
                    mask_slice,
                    fwhm_slice,
                    np.nan,
                )

                # Transpose so array axis 0 is displayed horizontally,
                # analogous to many MRSI map visualizations.
                metabolite_slice = metabolite_slice.T
                fwhm_slice = fwhm_slice.T

                metabolite_image = axes[
                    row,
                    0,
                ].imshow(
                    metabolite_slice,
                    origin="lower",
                    cmap="viridis",
                    vmin=metabolite_vmin,
                    vmax=metabolite_vmax,
                    interpolation="nearest",
                )

                fwhm_image = axes[
                    row,
                    1,
                ].imshow(
                    fwhm_slice,
                    origin="lower",
                    cmap="magma",
                    vmin=fwhm_vmin,
                    vmax=fwhm_vmax,
                    interpolation="nearest",
                )

                axes[
                    row,
                    0,
                ].set_title(
                    f"{matched_name} amplitude — slice {slice_index}"
                )

                axes[
                    row,
                    1,
                ].set_title(
                    f"FWHM — slice {slice_index}"
                )

                axes[
                    row,
                    0,
                ].set_xticks([])

                axes[
                    row,
                    0,
                ].set_yticks([])

                axes[
                    row,
                    1,
                ].set_xticks([])

                axes[
                    row,
                    1,
                ].set_yticks([])

            fig.suptitle(
                (
                    f"{subject_id}\n"
                    f"{matched_name} amplitude vs. FWHM"
                ),
                fontsize=14,
            )

            fig.tight_layout(
                rect=(
                    0,
                    0.04,
                    1,
                    0.97,
                )
            )

            if metabolite_image is not None:
                fig.colorbar(
                    metabolite_image,
                    ax=axes[:, 0],
                    fraction=0.025,
                    pad=0.02,
                    label=f"{matched_name} amplitude",
                )

            if fwhm_image is not None:
                fig.colorbar(
                    fwhm_image,
                    ax=axes[:, 1],
                    fraction=0.025,
                    pad=0.02,
                    label="FWHM",
                )

            if save_dir is not None:
                safe_subject_id = re.sub(
                    r"[^A-Za-z0-9_.-]+",
                    "_",
                    subject_id,
                )

                save_path = (
                    save_dir
                    / (
                        f"{safe_subject_id}_"
                        f"{matched_name}_FWHM_"
                        f"slices_{first_slice:02d}-"
                        f"{last_slice - 1:02d}.png"
                    )
                )

                fig.savefig(
                    save_path,
                    dpi=dpi,
                    bbox_inches="tight",
                )

            if show:
                plt.show()
            else:
                plt.close(
                    fig
                )

            subject_figures.append(
                fig
            )

        figures[
            subject_id
        ] = subject_figures

    return figures