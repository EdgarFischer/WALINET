from pathlib import Path

import numpy as np
import torch

from walinet.training_data.build_simulation_system import (
    build_simulation_system,
)


def simulate_validation_dataset(
    train_config_path: str | Path,
    *,
    n_spectra: int = 100_000,
    batch_size: int = 4096,
    seed: int = 12345,
) -> dict[str, object]:
    """
    Build the training simulator from one training configuration and
    generate a validation dataset for metabolite-ratio calibration.

    Returns
    -------
    metabolite_spectra:
        Noise-free simulated metabolite spectra after broadening,
        frequency shift and phase application.

        Shape:
            (n_spectra, n_timepoints)

        Dtype:
            complex64

    coefficients:
        Sampled metabolite coefficients aligned with ``basis_names``.

        Shape:
            (n_spectra, n_basis_components)

        Dtype:
            float32

    basis_names:
        Basis-component names corresponding to the columns of
        ``coefficients``.
    """
    n_spectra = int(
        n_spectra
    )
    batch_size = int(
        batch_size
    )
    seed = int(
        seed
    )

    if n_spectra <= 0:
        raise ValueError(
            "n_spectra must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    # Build the complete simulator from the training YAML.
    system = build_simulation_system(
        train_config_path
    )

    # We validate the distribution used during training.
    simulator = system.train_simulator

    basis_names = tuple(
        str(name)
        for name in system.prepared_basis.names
    )

    n_timepoints = int(
        system.prepared_basis.n_timepoints
    )

    n_basis_components = len(
        basis_names
    )

    # Generator must live on the same device as the simulator.
    generator = torch.Generator(
        device=system.device
    )

    generator.manual_seed(
        seed
    )

    # Preallocate CPU arrays to avoid keeping all batches on the GPU
    # or concatenating large temporary arrays afterwards.
    metabolite_spectra = np.empty(
        (
            n_spectra,
            n_timepoints,
        ),
        dtype=np.complex64,
    )

    coefficients = np.empty(
        (
            n_spectra,
            n_basis_components,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n_spectra,
        batch_size,
    ):
        stop = min(
            start + batch_size,
            n_spectra,
        )

        current_batch_size = (
            stop - start
        )

        # Use the same simulation entry point as during training.
        batch = simulator.simulate(
            batch_size=current_batch_size,
            generator=generator,
        )

        simulated_metabolites = (
            batch.raw.metabolites
        )

        spectra_batch = (
            simulated_metabolites
            .clean_spectra
            .detach()
            .cpu()
            .numpy()
        )

        coefficients_batch = (
            simulated_metabolites
            .concentrations
            .detach()
            .cpu()
            .numpy()
        )

        expected_spectra_shape = (
            current_batch_size,
            n_timepoints,
        )

        expected_coefficients_shape = (
            current_batch_size,
            n_basis_components,
        )

        if spectra_batch.shape != expected_spectra_shape:
            raise RuntimeError(
                "Unexpected simulated spectrum shape:\n"
                f"  expected: {expected_spectra_shape}\n"
                f"  found:    {spectra_batch.shape}"
            )

        if coefficients_batch.shape != expected_coefficients_shape:
            raise RuntimeError(
                "Unexpected coefficient shape:\n"
                f"  expected: {expected_coefficients_shape}\n"
                f"  found:    {coefficients_batch.shape}"
            )

        metabolite_spectra[
            start:stop
        ] = spectra_batch.astype(
            np.complex64,
            copy=False,
        )

        coefficients[
            start:stop
        ] = coefficients_batch.astype(
            np.float32,
            copy=False,
        )

    if not np.isfinite(
        metabolite_spectra
    ).all():
        raise RuntimeError(
            "The simulated metabolite spectra contain "
            "non-finite values."
        )

    if not np.isfinite(
        coefficients
    ).all():
        raise RuntimeError(
            "The simulated coefficients contain "
            "non-finite values."
        )

    print(
        "Validation simulation completed"
    )

    print(
        f"  spectra:      "
        f"{metabolite_spectra.shape} "
        f"{metabolite_spectra.dtype}"
    )

    print(
        f"  coefficients: "
        f"{coefficients.shape} "
        f"{coefficients.dtype}"
    )

    print(
        f"  basis names:  "
        f"{len(basis_names)}"
    )

    print(
        f"  device:       "
        f"{system.device}"
    )

    print(
        f"  seed:         "
        f"{seed}"
    )

    return {
        "metabolite_spectra": metabolite_spectra,
        "coefficients": coefficients,
        "basis_names": basis_names,
        "seed": seed,
        "train_config_path": str(
            Path(
                train_config_path
            )
            .expanduser()
            .resolve()
        ),
    }

import numpy as np


def calculate_simulated_r_values(
    simulation_validation: dict[str, object],
) -> dict[str, object]:
    """
    Calculate normalized metabolite coefficients from a previously
    simulated validation dataset.

    For every simulated spectrum:

        r_i = c_i / max(abs(S_metabolites))

    Parameters
    ----------
    simulation_validation:
        Output of ``simulate_validation_dataset``.

    Returns
    -------
    r_values:
        Normalized metabolite coefficients.

        Shape:
            (n_spectra, n_basis_components)

    normalization_scale:
        Maximum absolute metabolite signal for every spectrum.

        Shape:
            (n_spectra, 1)

    coefficients:
        Original sampled metabolite coefficients.

    basis_names:
        Names corresponding to the columns of ``r_values`` and
        ``coefficients``.
    """
    required_keys = {
        "metabolite_spectra",
        "coefficients",
        "basis_names",
    }

    missing_keys = sorted(
        required_keys.difference(
            simulation_validation
        )
    )

    if missing_keys:
        raise KeyError(
            "simulation_validation is missing entries:\n  "
            + "\n  ".join(missing_keys)
        )

    metabolite_spectra = np.asarray(
        simulation_validation[
            "metabolite_spectra"
        ]
    )

    coefficients = np.asarray(
        simulation_validation[
            "coefficients"
        ],
        dtype=np.float32,
    )

    basis_names = tuple(
        str(name)
        for name in simulation_validation[
            "basis_names"
        ]
    )

    # ---------------------------------------------------------
    # Validate dimensions
    # ---------------------------------------------------------
    if metabolite_spectra.ndim != 2:
        raise ValueError(
            "metabolite_spectra must have shape "
            "(n_spectra, n_timepoints), "
            f"but found {metabolite_spectra.shape}."
        )

    if coefficients.ndim != 2:
        raise ValueError(
            "coefficients must have shape "
            "(n_spectra, n_basis_components), "
            f"but found {coefficients.shape}."
        )

    if (
        metabolite_spectra.shape[0]
        != coefficients.shape[0]
    ):
        raise ValueError(
            "metabolite_spectra and coefficients contain "
            "different numbers of spectra.\n"
            f"  metabolite_spectra: "
            f"{metabolite_spectra.shape}\n"
            f"  coefficients:       "
            f"{coefficients.shape}"
        )

    if coefficients.shape[1] != len(
        basis_names
    ):
        raise ValueError(
            "The number of coefficient columns does not match "
            "the number of basis names.\n"
            f"  coefficient columns: "
            f"{coefficients.shape[1]}\n"
            f"  basis names:         "
            f"{len(basis_names)}"
        )

    if not np.isfinite(
        metabolite_spectra
    ).all():
        raise ValueError(
            "metabolite_spectra contains non-finite values."
        )

    if not np.isfinite(
        coefficients
    ).all():
        raise ValueError(
            "coefficients contains non-finite values."
        )

    # ---------------------------------------------------------
    # Same normalization as for the in-vivo calibration
    # ---------------------------------------------------------
    normalization_scale = np.max(
        np.abs(
            metabolite_spectra
        ),
        axis=-1,
        keepdims=True,
    ).astype(
        np.float32,
        copy=False,
    )

    invalid_scale = (
        ~np.isfinite(
            normalization_scale
        )
        | (
            normalization_scale <= 0
        )
    )

    if np.any(
        invalid_scale
    ):
        invalid_count = int(
            np.count_nonzero(
                invalid_scale
            )
        )

        raise ValueError(
            "Some simulated spectra have an invalid "
            "normalization scale.\n"
            f"  invalid spectra: {invalid_count}"
        )

    r_values = (
        coefficients
        / normalization_scale
    ).astype(
        np.float32,
        copy=False,
    )

    if not np.isfinite(
        r_values
    ).all():
        raise RuntimeError(
            "The calculated r_values contain "
            "non-finite values."
        )

    print(
        "Simulated metabolite ratios calculated"
    )

    print(
        f"  spectra:            "
        f"{metabolite_spectra.shape}"
    )

    print(
        f"  coefficients:       "
        f"{coefficients.shape}"
    )

    print(
        f"  normalization scale:"
        f" {normalization_scale.shape}"
    )

    print(
        f"  r values:           "
        f"{r_values.shape}"
    )

    print(
        f"  scale range:        "
        f"{normalization_scale.min():.6g} "
        f"to "
        f"{normalization_scale.max():.6g}"
    )

    return {
        "r_values": r_values,
        "coefficients": coefficients,
        "normalization_scale": normalization_scale,
        "basis_names": basis_names,
    }

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import yaml

from scipy.stats import lognorm, norm, truncnorm

from walinet.config.build import build_config
from walinet.config.build_simulation import (
    build_simulation_config,
)


def plot_metabolite_ratio_validation(
    *,
    train_config_path: str | Path,
    simulated_r_calibration: dict[str, object],
    metabolite_name: str,
    bins: int | str = 50,
    plot_percentile: float = 99.5,
    x_limits: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Probability density",
    save_path: str | Path | None = None,
    dpi: int = 300,
    n_model_points: int = 1200,
    show: bool = True,
) -> dict[str, object]:
    """
    Compare the configured metabolite-coefficient distribution with
    the normalized r_i distribution obtained after full simulation.

    The configured input distribution is read from the metabolite
    profile referenced by the training/simulation configuration:

        0.5 * truncated Normal
        +
        0.5 * LogNormal

    The simulated output values are taken from:

        simulated_r_calibration["r_values"]

    where:

        r_i = coefficient_i / max(abs(metabolite_spectrum))

    Multiple metabolite profiles are supported. Their configured
    distributions are combined according to their profile
    probabilities.
    """

    # ---------------------------------------------------------
    # Small helpers
    # ---------------------------------------------------------
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

    def load_yaml_mapping(
        path: str | Path,
    ) -> dict:
        path = Path(
            path
        ).expanduser().resolve()

        if not path.is_file():
            raise FileNotFoundError(
                "Configuration file not found:\n"
                f"  {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            raw = yaml.safe_load(
                file
            )

        if not isinstance(
            raw,
            dict,
        ):
            raise TypeError(
                "Configuration file must contain a YAML mapping:\n"
                f"  file:  {path}\n"
                f"  found: {type(raw)}"
            )

        return raw

    def positive_truncated_normal_pdf(
        x: np.ndarray,
        *,
        mean: float,
        std: float,
        minimum: float,
    ) -> np.ndarray:
        if std == 0:
            return np.zeros_like(
                x,
                dtype=np.float64,
            )

        standardized_minimum = (
            minimum - mean
        ) / std

        survival = norm.sf(
            standardized_minimum
        )

        if (
            not np.isfinite(survival)
            or survival <= 0
        ):
            raise ValueError(
                "Configured truncated-normal component has "
                "numerically zero probability above its minimum."
            )

        density = (
            norm.pdf(
                x,
                loc=mean,
                scale=std,
            )
            / survival
        )

        return np.where(
            x > minimum,
            density,
            0.0,
        )

    def positive_truncated_normal_cdf(
        x: np.ndarray,
        *,
        mean: float,
        std: float,
        minimum: float,
    ) -> np.ndarray:
        if std == 0:
            return (
                x >= mean
            ).astype(
                np.float64
            )

        standardized_minimum = (
            minimum - mean
        ) / std

        lower_cdf = norm.cdf(
            standardized_minimum
        )

        survival = norm.sf(
            standardized_minimum
        )

        cdf = (
            norm.cdf(
                x,
                loc=mean,
                scale=std,
            )
            - lower_cdf
        ) / survival

        return np.where(
            x <= minimum,
            0.0,
            np.clip(
                cdf,
                0.0,
                1.0,
            ),
        )

    def positive_truncated_lognormal_pdf(
        x: np.ndarray,
        *,
        log_mu: float,
        log_sigma: float,
        minimum: float,
    ) -> np.ndarray:
        if log_sigma == 0:
            return np.zeros_like(
                x,
                dtype=np.float64,
            )

        scale = float(
            np.exp(
                log_mu
            )
        )

        survival = lognorm.sf(
            minimum,
            s=log_sigma,
            scale=scale,
        )

        if (
            not np.isfinite(survival)
            or survival <= 0
        ):
            raise ValueError(
                "Configured lognormal component has numerically "
                "zero probability above its minimum."
            )

        density = (
            lognorm.pdf(
                x,
                s=log_sigma,
                scale=scale,
            )
            / survival
        )

        return np.where(
            x > minimum,
            density,
            0.0,
        )

    def positive_truncated_lognormal_cdf(
        x: np.ndarray,
        *,
        log_mu: float,
        log_sigma: float,
        minimum: float,
    ) -> np.ndarray:
        constant_value = float(
            np.exp(
                log_mu
            )
        )

        if log_sigma == 0:
            return (
                x >= constant_value
            ).astype(
                np.float64
            )

        scale = constant_value

        lower_cdf = lognorm.cdf(
            minimum,
            s=log_sigma,
            scale=scale,
        )

        survival = lognorm.sf(
            minimum,
            s=log_sigma,
            scale=scale,
        )

        cdf = (
            lognorm.cdf(
                x,
                s=log_sigma,
                scale=scale,
            )
            - lower_cdf
        ) / survival

        return np.where(
            x <= minimum,
            0.0,
            np.clip(
                cdf,
                0.0,
                1.0,
            ),
        )

    # ---------------------------------------------------------
    # Validate simulated results
    # ---------------------------------------------------------
    required_result_keys = {
        "r_values",
        "basis_names",
    }

    missing_result_keys = sorted(
        required_result_keys.difference(
            simulated_r_calibration
        )
    )

    if missing_result_keys:
        raise KeyError(
            "simulated_r_calibration is missing entries:\n  "
            + "\n  ".join(
                missing_result_keys
            )
        )

    r_values = np.asarray(
        simulated_r_calibration[
            "r_values"
        ],
        dtype=np.float64,
    )

    basis_names = tuple(
        str(name)
        for name in simulated_r_calibration[
            "basis_names"
        ]
    )

    if r_values.ndim != 2:
        raise ValueError(
            "r_values must have shape "
            "(n_spectra, n_basis_components), "
            f"but found {r_values.shape}."
        )

    if r_values.shape[1] != len(
        basis_names
    ):
        raise ValueError(
            "The number of r-value columns does not match "
            "the number of basis names."
        )

    target_key = name_key(
        metabolite_name
    )

    matching_basis_indices = [
        index
        for index, basis_name in enumerate(
            basis_names
        )
        if name_key(
            basis_name
        ) == target_key
    ]

    if not matching_basis_indices:
        available = "\n  ".join(
            basis_names
        )

        raise KeyError(
            f"Metabolite {metabolite_name!r} was not found "
            "in basis_names.\n"
            f"Available basis names:\n  {available}"
        )

    if len(
        matching_basis_indices
    ) > 1:
        raise ValueError(
            f"Metabolite name {metabolite_name!r} is ambiguous."
        )

    basis_index = matching_basis_indices[
        0
    ]

    basis_name = basis_names[
        basis_index
    ]

    all_simulated_values = r_values[
        :,
        basis_index,
    ]

    simulated_values = all_simulated_values[
        np.isfinite(
            all_simulated_values
        )
        & (
            all_simulated_values > 0
        )
    ]

    if simulated_values.size == 0:
        raise ValueError(
            f"No positive finite simulated r_i values were found "
            f"for {basis_name!r}."
        )

    # ---------------------------------------------------------
    # Load training and simulation configurations
    # ---------------------------------------------------------
    train_config_path = Path(
        train_config_path
    ).expanduser().resolve()

    train_raw = load_yaml_mapping(
        train_config_path
    )

    train_cfg = build_config(
        train_raw,
        config_dir=train_config_path.parent,
    )

    simulation_config_path = Path(
        train_cfg.data.simulation_config
    ).expanduser().resolve()

    simulation_raw = load_yaml_mapping(
        simulation_config_path
    )

    simulation_cfg = build_simulation_config(
        simulation_raw,
        config_dir=simulation_config_path.parent,
    )

    # ---------------------------------------------------------
    # Read configured metabolite distributions from all profiles
    # ---------------------------------------------------------
    configured_profiles = []

    for profile_cfg in (
        simulation_cfg.metabolites.profiles
    ):
        profile_path = Path(
            profile_cfg.config
        ).expanduser().resolve()

        profile_raw = load_yaml_mapping(
            profile_path
        )

        sampling_raw = profile_raw.get(
            "sampling",
            {},
        )

        if not isinstance(
            sampling_raw,
            dict,
        ):
            raise TypeError(
                f"{profile_path}: sampling must be a mapping."
            )

        default_distribution = str(
            sampling_raw.get(
                "default_distribution",
                "positive_mixture",
            )
        ).strip().lower()

        metabolites_raw = profile_raw.get(
            "metabolites"
        )

        if not isinstance(
            metabolites_raw,
            dict,
        ):
            raise TypeError(
                f"{profile_path}: metabolites must be a mapping."
            )

        matching_entries = []

        for (
            config_name,
            metabolite_raw,
        ) in metabolites_raw.items():
            if not isinstance(
                metabolite_raw,
                dict,
            ):
                continue

            basis_component = str(
                metabolite_raw.get(
                    "basis_component",
                    config_name,
                )
            ).strip()

            if (
                name_key(
                    config_name
                )
                == target_key
                or name_key(
                    basis_component
                )
                == target_key
            ):
                matching_entries.append(
                    (
                        str(
                            config_name
                        ),
                        basis_component,
                        metabolite_raw,
                    )
                )

        if len(
            matching_entries
        ) > 1:
            raise ValueError(
                f"{profile_path}: multiple metabolite entries match "
                f"{metabolite_name!r}."
            )

        if not matching_entries:
            continue

        (
            config_name,
            basis_component,
            metabolite_raw,
        ) = matching_entries[0]

        enabled = bool(
            metabolite_raw.get(
                "enabled",
                True,
            )
        )

        # Disabled components become exactly zero in this profile.
        # Since the histogram contains only positive values, this
        # profile is not part of the conditional positive density.
        if not enabled:
            continue

        distribution_raw = metabolite_raw.get(
            "distribution"
        )

        if not isinstance(
            distribution_raw,
            dict,
        ):
            raise TypeError(
                f"{profile_path}: distribution for "
                f"{config_name!r} must be a mapping."
            )

        distribution_type = str(
            distribution_raw.get(
                "type",
                default_distribution,
            )
        ).strip().lower()

        if distribution_type != "positive_mixture":
            raise ValueError(
                f"{profile_path}: expected positive_mixture for "
                f"{config_name!r}, but found "
                f"{distribution_type!r}."
            )

        normal_raw = distribution_raw.get(
            "normal"
        )

        lognormal_raw = distribution_raw.get(
            "lognormal"
        )

        if not isinstance(
            normal_raw,
            dict,
        ):
            raise TypeError(
                f"{profile_path}: normal distribution entry "
                "must be a mapping."
            )

        if not isinstance(
            lognormal_raw,
            dict,
        ):
            raise TypeError(
                f"{profile_path}: lognormal distribution entry "
                "must be a mapping."
            )

        normal_mean = float(
            normal_raw[
                "mean"
            ]
        )

        normal_std = float(
            normal_raw[
                "std"
            ]
        )

        log_mu = float(
            lognormal_raw[
                "log_mu"
            ]
        )

        log_sigma = float(
            lognormal_raw[
                "log_sigma"
            ]
        )

        minimum = float(
            distribution_raw[
                "minimum"
            ]
        )

        values_to_check = {
            "normal mean": normal_mean,
            "normal std": normal_std,
            "log_mu": log_mu,
            "log_sigma": log_sigma,
            "minimum": minimum,
        }

        for parameter_name, parameter_value in (
            values_to_check.items()
        ):
            if not np.isfinite(
                parameter_value
            ):
                raise ValueError(
                    f"{profile_path}: {parameter_name} for "
                    f"{config_name!r} is not finite."
                )

        if normal_std < 0:
            raise ValueError(
                f"{profile_path}: normal std must be >= 0."
            )

        if log_sigma < 0:
            raise ValueError(
                f"{profile_path}: log_sigma must be >= 0."
            )

        if minimum < 0:
            raise ValueError(
                f"{profile_path}: minimum must be >= 0."
            )

        configured_profiles.append(
            {
                "profile_path": str(
                    profile_path
                ),
                "profile_probability": float(
                    profile_cfg.probability
                ),
                "config_name": config_name,
                "basis_component": basis_component,
                "normal_mean": normal_mean,
                "normal_std": normal_std,
                "log_mu": log_mu,
                "log_sigma": log_sigma,
                "minimum": minimum,
            }
        )

    if not configured_profiles:
        raise ValueError(
            f"No enabled configured distribution was found for "
            f"{metabolite_name!r}."
        )

    # Profiles in which the metabolite is disabled produce zero.
    # Because we compare only positive values, renormalize the
    # probabilities across profiles where the metabolite is enabled.
    active_probability_sum = sum(
        profile[
            "profile_probability"
        ]
        for profile in configured_profiles
    )

    if active_probability_sum <= 0:
        raise ValueError(
            "The active profile probabilities sum to zero."
        )

    for profile in configured_profiles:
        profile[
            "conditional_probability"
        ] = (
            profile[
                "profile_probability"
            ]
            / active_probability_sum
        )

    # ---------------------------------------------------------
    # Recalculate robust output statistics
    # ---------------------------------------------------------
    (
        simulated_q1,
        simulated_median,
        simulated_q3,
    ) = np.percentile(
        simulated_values,
        [
            25,
            50,
            75,
        ],
    )

    simulated_iqr = (
        simulated_q3
        - simulated_q1
    )

    simulated_normal_sigma = (
        simulated_iqr
        / 1.349
    )

    simulated_log_values = np.log(
        simulated_values
    )

    (
        simulated_log_q1,
        simulated_log_mu,
        simulated_log_q3,
    ) = np.percentile(
        simulated_log_values,
        [
            25,
            50,
            75,
        ],
    )

    simulated_log_iqr = (
        simulated_log_q3
        - simulated_log_q1
    )

    simulated_log_sigma = (
        simulated_log_iqr
        / 1.349
    )

    # ---------------------------------------------------------
    # Plotting range
    # ---------------------------------------------------------
    if not (
        0 < plot_percentile <= 100
    ):
        raise ValueError(
            "plot_percentile must be in (0, 100]."
        )

    if n_model_points < 2:
        raise ValueError(
            "n_model_points must be at least 2."
        )

    # if x_limits is None:
    #     simulated_xmax = float(
    #         np.percentile(
    #             simulated_values,
    #             plot_percentile,
    #         )
    #     )

    #     quantile_probability = min(
    #         plot_percentile / 100.0,
    #         1.0 - 1e-8,
    #     )

    #     configured_quantiles = []

    #     for profile in configured_profiles:
    #         mean = profile[
    #             "normal_mean"
    #         ]

    #         std = profile[
    #             "normal_std"
    #         ]

    #         log_mu = profile[
    #             "log_mu"
    #         ]

    #         log_sigma = profile[
    #             "log_sigma"
    #         ]

    #         minimum = profile[
    #             "minimum"
    #         ]

    #         if std > 0:
    #             normal_quantile = truncnorm.ppf(
    #                 quantile_probability,
    #                 a=(
    #                     minimum - mean
    #                 ) / std,
    #                 b=np.inf,
    #                 loc=mean,
    #                 scale=std,
    #             )
    #         else:
    #             normal_quantile = mean

    #         if log_sigma > 0:
    #             scale = float(
    #                 np.exp(
    #                     log_mu
    #                 )
    #             )

    #             lower_cdf = lognorm.cdf(
    #                 minimum,
    #                 s=log_sigma,
    #                 scale=scale,
    #             )

    #             target_cdf = (
    #                 lower_cdf
    #                 + quantile_probability
    #                 * (
    #                     1.0 - lower_cdf
    #                 )
    #             )

    #             lognormal_quantile = lognorm.ppf(
    #                 target_cdf,
    #                 s=log_sigma,
    #                 scale=scale,
    #             )
    #         else:
    #             lognormal_quantile = float(
    #                 np.exp(
    #                     log_mu
    #                 )
    #             )

    #         configured_quantiles.extend(
    #             [
    #                 normal_quantile,
    #                 lognormal_quantile,
    #             ]
    #         )

    #     finite_configured_quantiles = [
    #         float(value)
    #         for value in configured_quantiles
    #         if np.isfinite(
    #             value
    #         )
    #         and value > 0
    #     ]

    #     x_min = 0.0

    #     x_max = max(
    #         [
    #             simulated_xmax,
    #             *finite_configured_quantiles,
    #         ]
    #     )

    if x_limits is None:
        x_min = 0.0

        x_max = float(
            np.percentile(
                simulated_values,
                plot_percentile,
            )
        )
    else:
        x_min, x_max = map(
            float,
            x_limits,
        )

    if (
        not np.isfinite(
            x_min
        )
        or not np.isfinite(
            x_max
        )
        or x_min < 0
        or x_min >= x_max
    ):
        raise ValueError(
            "x limits must be finite and satisfy "
            "0 <= x_min < x_max."
        )

    model_x_min = max(
        x_min,
        x_max * 1e-8,
        np.finfo(
            np.float64
        ).tiny,
    )

    x = np.linspace(
        model_x_min,
        x_max,
        n_model_points,
    )

    # ---------------------------------------------------------
    # Construct the configured profile-weighted density
    # ---------------------------------------------------------
    configured_normal_pdf = np.zeros_like(
        x,
        dtype=np.float64,
    )

    configured_lognormal_pdf = np.zeros_like(
        x,
        dtype=np.float64,
    )

    configured_mixture_cdf = np.zeros_like(
        x,
        dtype=np.float64,
    )

    for profile in configured_profiles:
        weight = profile[
            "conditional_probability"
        ]

        profile_normal_pdf = (
            positive_truncated_normal_pdf(
                x,
                mean=profile[
                    "normal_mean"
                ],
                std=profile[
                    "normal_std"
                ],
                minimum=profile[
                    "minimum"
                ],
            )
        )

        profile_lognormal_pdf = (
            positive_truncated_lognormal_pdf(
                x,
                log_mu=profile[
                    "log_mu"
                ],
                log_sigma=profile[
                    "log_sigma"
                ],
                minimum=profile[
                    "minimum"
                ],
            )
        )

        profile_normal_cdf = (
            positive_truncated_normal_cdf(
                x,
                mean=profile[
                    "normal_mean"
                ],
                std=profile[
                    "normal_std"
                ],
                minimum=profile[
                    "minimum"
                ],
            )
        )

        profile_lognormal_cdf = (
            positive_truncated_lognormal_cdf(
                x,
                log_mu=profile[
                    "log_mu"
                ],
                log_sigma=profile[
                    "log_sigma"
                ],
                minimum=profile[
                    "minimum"
                ],
            )
        )

        configured_normal_pdf += (
            weight
            * profile_normal_pdf
        )

        configured_lognormal_pdf += (
            weight
            * profile_lognormal_pdf
        )

        configured_mixture_cdf += (
            weight
            * (
                0.5
                * profile_normal_cdf
                + 0.5
                * profile_lognormal_cdf
            )
        )

    configured_mixture_pdf = (
        0.5
        * configured_normal_pdf
        + 0.5
        * configured_lognormal_pdf
    )

    if configured_mixture_cdf[-1] >= 0.5:
        configured_median = float(
            np.interp(
                0.5,
                configured_mixture_cdf,
                x,
            )
        )
    else:
        configured_median = np.nan

    # ---------------------------------------------------------
    # Plot labels
    # ---------------------------------------------------------
    if title is None:
        title = (
            f"{basis_name} coefficient-ratio validation"
        )

    if xlabel is None:
        xlabel = (
            f"{basis_name} coefficient / "
            "max|Metabolites|"
        )

    if len(
        configured_profiles
    ) == 1:
        profile = configured_profiles[
            0
        ]

        normal_label = (
            "Configured zero-truncated normal component\n"
            f"μ={profile['normal_mean']:.3g}, "
            f"σ={profile['normal_std']:.3g}"
        )

        lognormal_label = (
            "Configured lognormal component\n"
            f"log-μ={profile['log_mu']:.3g}, "
            f"log-σ={profile['log_sigma']:.3g}"
        )

    else:
        normal_label = (
            "Configured zero-truncated normal component\n"
            f"{len(configured_profiles)} profile-weighted profiles"
        )

        lognormal_label = (
            "Configured lognormal component\n"
            f"{len(configured_profiles)} profile-weighted profiles"
        )

    # ---------------------------------------------------------
    # Same basic design as compare_positive_models
    # ---------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(
            9,
            5,
        )
    )

    ax.hist(
        simulated_values,
        bins=bins,
        range=(
            x_min,
            x_max,
        ),
        density=True,
        alpha=0.6,
        label="After simulator normalization",
    )

    ax.plot(
        x,
        configured_normal_pdf,
        linestyle="--",
        linewidth=1.5,
        label=normal_label,
    )

    ax.plot(
        x,
        configured_lognormal_pdf,
        linestyle=":",
        linewidth=1.8,
        label=lognormal_label,
    )

    ax.plot(
        x,
        configured_mixture_pdf,
        linewidth=3,
        label="Configured 50/50 mixture",
    )

    if np.isfinite(
        configured_median
    ):
        ax.axvline(
            configured_median,
            linestyle="--",
            linewidth=1,
            label=(
                f"Configured median="
                f"{configured_median:.3g}"
            ),
        )

    ax.axvline(
        simulated_median,
        linestyle="-.",
        linewidth=1,
        label=(
            f"After-simulation median="
            f"{simulated_median:.3g}"
        ),
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

    if show:
        plt.show()

    # ---------------------------------------------------------
    # Diagnostic output
    # ---------------------------------------------------------
    print(
        f"\n{basis_name} validation"
    )

    print(
        "="
        * (
            len(
                basis_name
            )
            + 11
        )
    )

    print(
        f"Positive simulated values: "
        f"{simulated_values.size}"
    )

    print(
        f"Zero/non-positive values:   "
        f"{all_simulated_values.size - simulated_values.size}"
    )

    print(
        f"Active config profiles:     "
        f"{len(configured_profiles)}"
    )

    if len(
        configured_profiles
    ) == 1:
        profile = configured_profiles[
            0
        ]

        print(
            "\nConfigured input distribution:"
        )

        print(
            f"  normal_mu       = "
            f"{profile['normal_mean']:.6f}"
        )

        print(
            f"  normal_sigma    = "
            f"{profile['normal_std']:.6f}"
        )

        print(
            f"  log_mu          = "
            f"{profile['log_mu']:.6f}"
        )

        print(
            f"  log_sigma       = "
            f"{profile['log_sigma']:.6f}"
        )

        print(
            f"  minimum         = "
            f"{profile['minimum']:.6f}"
        )

    else:
        print(
            "\nConfigured input profiles:"
        )

        for profile in configured_profiles:
            print(
                f"  {profile['profile_path']}"
            )

            print(
                f"    conditional weight = "
                f"{profile['conditional_probability']:.6f}"
            )

            print(
                f"    normal_mu         = "
                f"{profile['normal_mean']:.6f}"
            )

            print(
                f"    normal_sigma      = "
                f"{profile['normal_std']:.6f}"
            )

            print(
                f"    log_mu            = "
                f"{profile['log_mu']:.6f}"
            )

            print(
                f"    log_sigma         = "
                f"{profile['log_sigma']:.6f}"
            )

    print(
        "\nAfter simulator normalization:"
    )

    print(
        f"  q1              = "
        f"{simulated_q1:.6f}"
    )

    print(
        f"  median          = "
        f"{simulated_median:.6f}"
    )

    print(
        f"  q3              = "
        f"{simulated_q3:.6f}"
    )

    print(
        f"  IQR             = "
        f"{simulated_iqr:.6f}"
    )

    print(
        f"  robust_sigma    = "
        f"{simulated_normal_sigma:.6f}"
    )

    print(
        f"  log_mu          = "
        f"{simulated_log_mu:.6f}"
    )

    print(
        f"  robust_log_sigma = "
        f"{simulated_log_sigma:.6f}"
    )

    return {
        "metabolite_name": basis_name,
        "basis_index": basis_index,
        "configured_profiles": configured_profiles,
        "configured_median": configured_median,
        "simulated_values": simulated_values,
        "simulated_q1": float(
            simulated_q1
        ),
        "simulated_median": float(
            simulated_median
        ),
        "simulated_q3": float(
            simulated_q3
        ),
        "simulated_iqr": float(
            simulated_iqr
        ),
        "simulated_normal_sigma": float(
            simulated_normal_sigma
        ),
        "simulated_log_mu": float(
            simulated_log_mu
        ),
        "simulated_log_sigma": float(
            simulated_log_sigma
        ),
        "plot_xmin": float(
            x_min
        ),
        "plot_xmax": float(
            x_max
        ),
        "figure": fig,
        "axes": ax,
    }