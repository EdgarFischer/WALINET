from __future__ import annotations
from .acquisition import PreparedBasis

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import h5py

from .hlsvd import ProcessedLCModelBasis
from .parser import LCModelBasis

from .library import validate_basis_library

def plot_basis_before_after_grid(
    basis: LCModelBasis,
    processed_basis: ProcessedLCModelBasis,
    *,
    ppm_limits: tuple[float, float] = (4.5, 0.0),
    n_columns: int = 4,
) -> None:
    """
    Plot original and HLSVD-cleaned basis components in a grid.
    """
    if processed_basis.names != basis.names:
        raise ValueError(
            "Metabolite order differs between basis and processed basis."
        )

    original_spectra = np.fft.fftshift(
        np.fft.fft(
            processed_basis.original_fids,
            axis=-1,
        ),
        axes=-1,
    )

    clean_spectra = np.fft.fftshift(
        np.fft.fft(
            processed_basis.clean_fids,
            axis=-1,
        ),
        axes=-1,
    )

    residual_spectra = (
        original_spectra
        - clean_spectra
    )

    n_points = processed_basis.n_points
    spectral_width_hz = basis.bandwidth

    frequency_hz = np.linspace(
        -spectral_width_hz / 2
        + spectral_width_hz / (2 * n_points),
        spectral_width_hz / 2
        - spectral_width_hz / (2 * n_points),
        n_points,
    )

    ppm = (
        frequency_hz / basis.hz_per_ppm
        + processed_basis.ppm_reference
    )

    n_components = processed_basis.n_metabolites
    n_rows = math.ceil(n_components / n_columns)

    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(16, 3 * n_rows),
        sharex=True,
    )

    axes = np.asarray(axes).ravel()

    for ax, name, original, cleaned, residual in zip(
        axes,
        processed_basis.names,
        original_spectra,
        clean_spectra,
        residual_spectra,
    ):
        ax.plot(
            ppm,
            np.abs(original),
            label="Original",
        )

        ax.plot(
            ppm,
            np.abs(cleaned),
            linestyle="--",
            label="After HLSVD",
        )

        ax.plot(
            ppm,
            np.abs(residual),
            linestyle=":",
            label="Residual",
            alpha=0.8,
        )

        ax.set_title(name)
        ax.set_xlim(*ppm_limits)
        ax.grid(alpha=0.3)

    for ax in axes[n_components:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper right",
    )

    fig.supxlabel("Chemical shift [ppm]")
    fig.supylabel("Magnitude [a.u.]")
    fig.suptitle(
        "LCModel basis components before and after HLSVD",
        fontsize=16,
    )

    plt.tight_layout(
        rect=(0, 0, 0.97, 0.97)
    )

    plt.show()

def plot_basis_library_consistency(
    library_path: str | Path,
    *,
    component_names: list[str] | None = None,
    ppm_limits: tuple[float, float] = (4.5, 0.0),
    n_columns: int = 4,
    magnitude: bool = True,
    show_removed_reference: bool = True,
) -> None:
    """
    Load a saved WALINET LCModel basis library and plot all selected
    components before and after HLSVD reference removal.

    The function also checks whether

        original_fid == clean_fid + removed_reference_fid

    within numerical precision.

    Parameters
    ----------
    library_path:
        Path to the saved HDF5 basis library.

    component_names:
        Optional list of components to plot. When omitted, all stored
        components are plotted in alphabetical order.

    ppm_limits:
        Displayed ppm range. A descending tuple gives the conventional
        MRS orientation.

    n_columns:
        Number of subplot columns.

    magnitude:
        Plot spectral magnitude when True. Otherwise plot the real part.

    show_removed_reference:
        Also plot the removed HLSVD reference component.
    """
    library_path = Path(library_path)

    if not library_path.is_file():
        raise FileNotFoundError(
            f"Basis library does not exist: {library_path}"
        )

    with h5py.File(library_path, "r") as h5:
        validate_basis_library(h5)

        components_group = h5["components"]

        if component_names is None:
            names = sorted(components_group.keys())
        else:
            names = list(component_names)

        if not names:
            raise ValueError(
                "No basis components were selected."
            )

        missing_components = [
            name
            for name in names
            if name not in components_group
        ]

        if missing_components:
            raise KeyError(
                "Components not found in basis library: "
                + ", ".join(missing_components)
            )

        original_fids = []
        clean_fids = []
        reference_fids = []

        dwell_times = []
        bandwidths_hz = []
        hz_per_ppm_values = []
        ppm_references = []

        stored_reconstruction_errors = []

        for name in names:
            component = components_group[name]

            original_fids.append(
                component["original_fid"][...]
            )

            clean_fids.append(
                component["clean_fid"][...]
            )

            reference_fids.append(
                component["removed_reference_fid"][...]
            )

            dwell_times.append(
                float(component.attrs["dwell_time"])
            )

            bandwidths_hz.append(
                float(component.attrs["bandwidth_hz"])
            )

            hz_per_ppm_values.append(
                float(component.attrs["hz_per_ppm"])
            )

            ppm_references.append(
                float(component.attrs["ppm_reference"])
            )

            stored_reconstruction_errors.append(
                float(
                    component.attrs[
                        "maximum_reconstruction_error"
                    ]
                )
            )

    original_fids = np.stack(
        original_fids,
        axis=0,
    )

    clean_fids = np.stack(
        clean_fids,
        axis=0,
    )

    reference_fids = np.stack(
        reference_fids,
        axis=0,
    )

    reference_shape = original_fids[0].shape

    if not (
        original_fids.shape
        == clean_fids.shape
        == reference_fids.shape
    ):
        raise ValueError(
            "Original, cleaned, and removed-reference arrays "
            "do not have matching shapes."
        )

    if not all(
        fid.shape == reference_shape
        for fid in original_fids
    ):
        raise ValueError(
            "Stored basis components have inconsistent shapes."
        )

    if not np.allclose(
        dwell_times,
        dwell_times[0],
    ):
        raise ValueError(
            "Stored components have inconsistent dwell times."
        )

    if not np.allclose(
        bandwidths_hz,
        bandwidths_hz[0],
    ):
        raise ValueError(
            "Stored components have inconsistent bandwidths."
        )

    if not np.allclose(
        hz_per_ppm_values,
        hz_per_ppm_values[0],
    ):
        raise ValueError(
            "Stored components have inconsistent Hz/ppm values."
        )

    if not np.allclose(
        ppm_references,
        ppm_references[0],
    ):
        raise ValueError(
            "Stored components have inconsistent ppm references."
        )

    reconstruction_errors = np.max(
        np.abs(
            clean_fids
            + reference_fids
            - original_fids
        ),
        axis=-1,
    )

    maximum_reconstruction_error = float(
        np.max(reconstruction_errors)
    )

    print(f"Basis library: {library_path.resolve()}")
    print(f"Components: {len(names)}")
    print(f"FID shape: {original_fids.shape}")
    print(
        "Maximum reconstruction error after loading:",
        maximum_reconstruction_error,
    )
    print(
        "Maximum reconstruction error stored in metadata:",
        max(stored_reconstruction_errors),
    )

    original_spectra = np.fft.fftshift(
        np.fft.fft(
            original_fids,
            axis=-1,
        ),
        axes=-1,
    )

    clean_spectra = np.fft.fftshift(
        np.fft.fft(
            clean_fids,
            axis=-1,
        ),
        axes=-1,
    )

    reference_spectra = np.fft.fftshift(
        np.fft.fft(
            reference_fids,
            axis=-1,
        ),
        axes=-1,
    )

    n_points = original_fids.shape[-1]
    bandwidth_hz = bandwidths_hz[0]
    hz_per_ppm = hz_per_ppm_values[0]
    ppm_reference = ppm_references[0]

    frequency_hz = np.fft.fftshift(
        np.fft.fftfreq(
            n_points,
            d=dwell_times[0],
        )
    )

    ppm_axis = (
        frequency_hz / hz_per_ppm
        + ppm_reference
    )

    if magnitude:
        original_values = np.abs(
            original_spectra
        )
        clean_values = np.abs(
            clean_spectra
        )
        reference_values = np.abs(
            reference_spectra
        )
        y_label = "Magnitude [a.u.]"
    else:
        original_values = np.real(
            original_spectra
        )
        clean_values = np.real(
            clean_spectra
        )
        reference_values = np.real(
            reference_spectra
        )
        y_label = "Real spectrum [a.u.]"

    n_rows = math.ceil(
        len(names) / n_columns
    )

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.2 * n_columns,
            3.0 * n_rows,
        ),
        sharex=True,
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for index, name in enumerate(names):
        axis = axes[index]

        axis.plot(
            ppm_axis,
            original_values[index],
            label="Original",
        )

        axis.plot(
            ppm_axis,
            clean_values[index],
            linestyle="--",
            label="After HLSVD",
        )

        if show_removed_reference:
            axis.plot(
                ppm_axis,
                reference_values[index],
                linestyle=":",
                label="Removed reference",
            )

        axis.set_title(
            f"{name}\n"
            f"error={reconstruction_errors[index]:.2e}"
        )

        axis.set_xlim(
            ppm_limits
        )

        axis.grid(
            alpha=0.3
        )

    for axis in axes[len(names):]:
        axis.axis("off")

    handles, labels = (
        axes[0].get_legend_handles_labels()
    )

    figure.legend(
        handles,
        labels,
        loc="upper right",
    )

    figure.supxlabel(
        "Chemical shift [ppm]"
    )

    figure.supylabel(
        y_label
    )

    figure.suptitle(
        "Stored LCModel basis library consistency check",
        fontsize=15,
    )

    figure.tight_layout(
        rect=(0.02, 0.02, 0.97, 0.96)
    )

    plt.show()

def plot_prepared_basis_grid(
    prepared_basis: PreparedBasis,
    *,
    ppm_limits: tuple[float, float] = (5.5, -0.5),
    n_columns: int = 4,
    magnitude: bool = True,
    normalize_each: bool = False,
) -> None:
    """
    Plot all components of a basis after adaptation to the target
    acquisition bandwidth and number of time points.

    Parameters
    ----------
    prepared_basis:
        Basis returned by ``prepare_basis_for_acquisition``.

    ppm_limits:
        Displayed ppm interval. A descending tuple gives the usual
        MRS orientation.

    n_columns:
        Number of columns in the subplot grid.

    magnitude:
        Plot spectral magnitude when True. Otherwise plot the real part.

    normalize_each:
        Normalize every component independently within the displayed
        ppm interval. Useful for visual inspection of weak components.
        Keep False to preserve relative basis scaling.
    """
    fids = np.asarray(
        prepared_basis.fids,
    )

    if fids.ndim != 2:
        raise ValueError(
            "Prepared basis FIDs must have shape "
            "(n_metabolites, n_timepoints), "
            f"but found {fids.shape}."
        )

    expected_shape = (
        len(prepared_basis.names),
        prepared_basis.n_timepoints,
    )

    if fids.shape != expected_shape:
        raise ValueError(
            f"Prepared basis has shape {fids.shape}; "
            f"expected {expected_shape}."
        )

    if not np.all(np.isfinite(fids)):
        raise ValueError(
            "Prepared basis contains non-finite values."
        )

    bandwidth_from_dwell_time = (
        1.0 / prepared_basis.dwell_time
    )

    if not np.isclose(
        prepared_basis.bandwidth,
        bandwidth_from_dwell_time,
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError(
            "Prepared basis bandwidth and dwell time "
            "are inconsistent:\n"
            f"  bandwidth: {prepared_basis.bandwidth}\n"
            f"  1 / dwell_time: {bandwidth_from_dwell_time}"
        )

    spectra = np.fft.fftshift(
        np.fft.fft(
            fids,
            axis=-1,
        ),
        axes=-1,
    )

    frequency_hz = np.fft.fftshift(
        np.fft.fftfreq(
            prepared_basis.n_timepoints,
            d=prepared_basis.dwell_time,
        )
    )

    ppm_axis = (
        frequency_hz
        / prepared_basis.hz_per_ppm
        + prepared_basis.ppm_reference
    )

    if magnitude:
        plot_values = np.abs(
            spectra
        )
        y_label = "Magnitude [a.u.]"
    else:
        plot_values = np.real(
            spectra
        )
        y_label = "Real spectrum [a.u.]"

    lower_ppm = min(ppm_limits)
    upper_ppm = max(ppm_limits)

    displayed_mask = (
        (ppm_axis >= lower_ppm)
        & (ppm_axis <= upper_ppm)
    )

    if not np.any(displayed_mask):
        raise ValueError(
            "The selected ppm interval does not overlap "
            "with the prepared basis frequency range."
        )

    if normalize_each:
        normalization = np.max(
            np.abs(
                plot_values[
                    :,
                    displayed_mask,
                ]
            ),
            axis=-1,
            keepdims=True,
        )

        normalization[
            normalization == 0
        ] = 1.0

        plot_values = (
            plot_values / normalization
        )

        y_label = (
            "Individually normalized "
            + y_label
        )

    n_metabolites = len(
        prepared_basis.names
    )

    n_rows = math.ceil(
        n_metabolites / n_columns
    )

    figure, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(
            4.2 * n_columns,
            3.0 * n_rows,
        ),
        sharex=True,
    )

    axes = np.atleast_1d(
        axes
    ).ravel()

    for index, name in enumerate(
        prepared_basis.names
    ):
        axis = axes[index]

        axis.plot(
            ppm_axis,
            plot_values[index],
        )

        axis.set_title(name)

        axis.set_xlim(
            *ppm_limits
        )

        axis.grid(
            alpha=0.3
        )

    for axis in axes[n_metabolites:]:
        axis.axis("off")

    frequency_resolution = (
        prepared_basis.bandwidth
        / prepared_basis.n_timepoints
    )

    acquisition_duration = (
        prepared_basis.n_timepoints
        * prepared_basis.dwell_time
    )

    figure.supxlabel(
        "Chemical shift [ppm]"
    )

    figure.supylabel(
        y_label
    )

    figure.suptitle(
        "LCModel basis prepared for acquisition\n"
        f"{prepared_basis.n_timepoints} points, "
        f"{prepared_basis.bandwidth:.3f} Hz bandwidth, "
        f"{frequency_resolution:.3f} Hz/bin",
        fontsize=15,
    )

    figure.tight_layout(
        rect=(0.02, 0.02, 0.98, 0.94)
    )

    print("Prepared-basis consistency check")
    print(
        f"  Shape                : "
        f"{prepared_basis.fids.shape}"
    )
    print(
        f"  Dtype                : "
        f"{prepared_basis.fids.dtype}"
    )
    print(
        f"  Finite values        : "
        f"{np.all(np.isfinite(fids))}"
    )
    print(
        f"  C contiguous         : "
        f"{prepared_basis.fids.flags.c_contiguous}"
    )
    print(
        f"  Requested bandwidth  : "
        f"{prepared_basis.requested_bandwidth:.6f} Hz"
    )
    print(
        f"  Actual bandwidth     : "
        f"{prepared_basis.bandwidth:.6f} Hz"
    )
    print(
        f"  Dwell time           : "
        f"{prepared_basis.dwell_time:.9e} s"
    )
    print(
        f"  Acquisition duration : "
        f"{acquisition_duration:.6f} s"
    )
    print(
        f"  Frequency resolution : "
        f"{frequency_resolution:.6f} Hz"
    )
    print(
        f"  ppm range            : "
        f"{ppm_axis.min():.3f} to "
        f"{ppm_axis.max():.3f}"
    )

    plt.show()