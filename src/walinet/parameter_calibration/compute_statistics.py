from __future__ import annotations

import numpy as np


def extract_valid_voxels(
    maps: np.ndarray,
    quality_mask: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """
    Extract valid voxel values separately for each subject and pool
    them across all subjects.

    Parameters
    ----------
    maps:
        Parameter maps with shape:

            (x, y, z, n_subjects)

    quality_mask:
        Binary quality mask with the same shape as ``maps``.
        Voxels with value 1 are included; voxels with value 0 are
        excluded.

    Returns
    -------
    subject_values:
        Dictionary containing one one-dimensional NumPy array per
        subject. The keys are strings:

            {
                "0": values_subject_0,
                "1": values_subject_1,
                ...
            }

    pooled_values:
        One-dimensional NumPy array containing all valid voxel values
        from all subjects, concatenated in subject order.
    """
    maps = np.asarray(maps)
    quality_mask = np.asarray(quality_mask)

    if maps.ndim != 4:
        raise ValueError(
            "maps must have shape (x, y, z, n_subjects), "
            f"but found shape {maps.shape}."
        )

    if quality_mask.shape != maps.shape:
        raise ValueError(
            "maps and quality_mask must have the same shape:\n"
            f"  maps:         {maps.shape}\n"
            f"  quality_mask: {quality_mask.shape}"
        )

    mask_is_finite = np.isfinite(quality_mask)

    unique_mask_values = np.unique(
        quality_mask[mask_is_finite]
    )

    if not np.all(
        np.isin(
            unique_mask_values,
            [0, 1],
        )
    ):
        raise ValueError(
            "quality_mask may contain only 0 and 1, "
            f"but found values {unique_mask_values}."
        )

    subject_values: dict[str, np.ndarray] = {}

    n_subjects = maps.shape[-1]

    for subject_index in range(n_subjects):
        subject_map = maps[..., subject_index]
        subject_mask = quality_mask[..., subject_index]

        valid = (
            (subject_mask == 1)
            & np.isfinite(subject_map)
        )

        values = subject_map[valid].reshape(-1)

        subject_values[str(subject_index)] = values

    nonempty_arrays = [
        values
        for values in subject_values.values()
        if values.size > 0
    ]

    if nonempty_arrays:
        pooled_values = np.concatenate(
            nonempty_arrays,
            axis=0,
        )
    else:
        pooled_values = np.empty(
            0,
            dtype=maps.dtype,
        )

    return subject_values, pooled_values

import numpy as np


def calculate_pooled_median_iqr(
    pooled_values: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate the median and interquartile range of pooled voxel values.

    Parameters
    ----------
    pooled_values:
        One-dimensional array containing pooled voxel values from all
        subjects.

    Returns
    -------
    median:
        Median of all pooled values.

    iqr:
        Interquartile range:

            IQR = Q3 - Q1
    """
    pooled_values = np.asarray(
        pooled_values,
        dtype=np.float64,
    )

    if pooled_values.ndim != 1:
        raise ValueError(
            "pooled_values must be one-dimensional, "
            f"but found shape {pooled_values.shape}."
        )

    pooled_values = pooled_values[
        np.isfinite(pooled_values)
    ]

    if pooled_values.size == 0:
        raise ValueError(
            "pooled_values contains no finite values."
        )

    q1, median, q3 = np.percentile(
        pooled_values,
        [25, 50, 75],
    )

    iqr = q3 - q1

    return float(median), float(iqr)