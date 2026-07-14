from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

from .library import (
    load_basis_fids,
    validate_basis_library,
)


BasisDatasetName = Literal[
    "original_fid",
    "clean_fid",
    "removed_reference_fid",
]


@dataclass
class PreparedBasis:
    """
    LCModel basis prepared for one acquisition protocol.

    The basis FIDs are stored in one contiguous array with shape

        (n_metabolites, n_timepoints)

    and can therefore be combined efficiently using matrix
    multiplication.
    """

    names: list[str]
    fids: np.ndarray

    bandwidth: float
    dwell_time: float
    n_timepoints: int

    hz_per_ppm: float
    ppm_reference: float

    requested_bandwidth: float

    source_bandwidth: float
    source_dwell_time: float
    source_n_points: int

    dataset_name: str
    library_path: Path

    @property
    def n_metabolites(self) -> int:
        return self.fids.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        return self.fids.shape

    @property
    def time_axis(self) -> np.ndarray:
        return (
            np.arange(
                self.n_timepoints,
                dtype=np.float64,
            )
            * self.dwell_time
        )

    def index(self, name: str) -> int:
        """
        Return the array index of one basis component.
        """
        try:
            return self.names.index(name)

        except ValueError as error:
            available = ", ".join(self.names)

            raise KeyError(
                f"Basis component not found: {name!r}\n"
                f"Available components: {available}"
            ) from error

    def as_dict(self) -> dict[str, np.ndarray]:
        """
        Return a dictionary view of the prepared FIDs.

        This is convenient for inspection, but the stacked `fids`
        array should be used for simulations because it is faster.
        """
        return {
            name: self.fids[index]
            for index, name in enumerate(
                self.names
            )
        }


def _read_common_ppm_reference(
    library_path: Path,
    component_names: list[str],
) -> float:
    """
    Read and validate the ppm reference stored for all components.
    """
    ppm_references: list[float] = []

    with h5py.File(
        library_path,
        "r",
    ) as h5:
        validate_basis_library(h5)

        components = h5["components"]

        for name in component_names:
            component = components[name]

            if "ppm_reference" not in component.attrs:
                raise ValueError(
                    f"Component {name!r} has no stored "
                    "ppm_reference metadata."
                )

            ppm_references.append(
                float(
                    component.attrs[
                        "ppm_reference"
                    ]
                )
            )

    reference = ppm_references[0]

    if not np.allclose(
        ppm_references,
        reference,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError(
            "The selected basis components have "
            "inconsistent ppm references."
        )

    return reference


def _choose_resampled_length(
    *,
    source_n_points: int,
    source_frequency_resolution: float,
    target_bandwidth: float,
) -> int:
    """
    Choose a centered FFT length corresponding as closely as possible
    to the requested bandwidth.

    The parity is kept equal to the native FFT length so that the
    zero-frequency bin remains centered exactly during cropping or
    zero-padding.
    """
    approximate_length = int(
        round(
            target_bandwidth
            / source_frequency_resolution
        )
    )

    if approximate_length < 2:
        raise ValueError(
            "Target bandwidth is too small for the "
            "native frequency resolution."
        )

    required_parity = (
        source_n_points % 2
    )

    if (
        approximate_length % 2
        != required_parity
    ):
        candidates = [
            value
            for value in (
                approximate_length - 1,
                approximate_length + 1,
            )
            if value >= 2
            and value % 2
            == required_parity
        ]

        approximate_length = min(
            candidates,
            key=lambda value: abs(
                value
                * source_frequency_resolution
                - target_bandwidth
            ),
        )

    return approximate_length


def _resize_centered_spectra(
    spectra: np.ndarray,
    *,
    target_n_points: int,
) -> np.ndarray:
    """
    Center-crop or center-pad spectra along the last dimension.

    Both the native and target lengths must have equal parity.
    """
    spectra = np.asarray(spectra)

    source_n_points = spectra.shape[-1]

    if (
        source_n_points % 2
        != target_n_points % 2
    ):
        raise ValueError(
            "Source and target FFT lengths must "
            "have equal parity."
        )

    if target_n_points == source_n_points:
        return spectra.copy()

    if target_n_points < source_n_points:
        start = (
            source_n_points
            - target_n_points
        ) // 2

        stop = start + target_n_points

        return spectra[..., start:stop]

    padding_total = (
        target_n_points
        - source_n_points
    )

    padding_left = padding_total // 2
    padding_right = (
        padding_total
        - padding_left
    )

    padding = [
        (0, 0)
        for _ in range(
            spectra.ndim
        )
    ]

    padding[-1] = (
        padding_left,
        padding_right,
    )

    return np.pad(
        spectra,
        padding,
        mode="constant",
    )


def prepare_basis_for_acquisition(
    library_path: str | Path,
    *,
    target_bandwidth: float,
    target_n_timepoints: int,
    component_names: list[str] | None = None,
    dataset_name: BasisDatasetName = "clean_fid",
    output_dtype: np.dtype = np.complex64,
    max_relative_bandwidth_error: float = 1e-3,
    verbose: bool = True,
) -> PreparedBasis:
    """
    Load a native WALINET LCModel basis library and prepare its FIDs
    for a target acquisition protocol.

    Parameters
    ----------
    library_path:
        Path to the WALINET HDF5 basis library.

    target_bandwidth:
        Requested acquisition bandwidth in Hz.

    target_n_timepoints:
        Number of output FID samples.

    component_names:
        Optional ordered list of basis components. If omitted, all
        components are loaded alphabetically.

        The supplied order is preserved and later determines the
        ordering expected for concentration arrays.

    dataset_name:
        Stored HDF5 signal to use. For simulation this should normally
        be ``"clean_fid"``.

    output_dtype:
        Complex dtype of the prepared basis. ``complex64`` is normally
        sufficient and efficient for training-data simulation.

    max_relative_bandwidth_error:
        Maximum allowed difference between requested bandwidth and the
        bandwidth achievable on the native frequency grid.

    verbose:
        Print a preparation summary.

    Returns
    -------
    PreparedBasis
        Prepared, contiguous basis array with shape

            (n_metabolites, target_n_timepoints)

    Notes
    -----
    Preparation is performed by:

    1. FFT of all native FIDs.
    2. Centered spectral cropping or zero-padding.
    3. Amplitude-preserving inverse FFT.
    4. Retention of the requested number of initial FID samples.

    The expensive preparation is performed only once. Subsequent
    simulations can combine all metabolites using matrix
    multiplication.
    """
    library_path = Path(
        library_path
    ).resolve()

    if not library_path.is_file():
        raise FileNotFoundError(
            f"Basis library does not exist: "
            f"{library_path}"
        )

    if not np.isfinite(
        target_bandwidth
    ) or target_bandwidth <= 0:
        raise ValueError(
            "target_bandwidth must be a "
            "positive finite number."
        )

    if (
        not isinstance(
            target_n_timepoints,
            int,
        )
        or target_n_timepoints <= 0
    ):
        raise ValueError(
            "target_n_timepoints must be "
            "a positive integer."
        )

    if max_relative_bandwidth_error < 0:
        raise ValueError(
            "max_relative_bandwidth_error "
            "must be non-negative."
        )

    if component_names is not None:
        if len(component_names) != len(
            set(component_names)
        ):
            raise ValueError(
                "component_names contains "
                "duplicate entries."
            )

    names, native_fids, metadata = (
        load_basis_fids(
            library_path,
            component_names=component_names,
            dataset_name=dataset_name,
        )
    )

    native_fids = np.asarray(
        native_fids,
    )

    if native_fids.ndim != 2:
        raise ValueError(
            "Expected native basis FIDs with "
            "shape (n_metabolites, n_points), "
            f"but found {native_fids.shape}."
        )

    if not np.all(
        np.isfinite(
            native_fids
        )
    ):
        raise ValueError(
            "Native basis FIDs contain "
            "non-finite values."
        )

    source_dwell_time = float(
        metadata["dwell_time"]
    )

    source_bandwidth_stored = float(
        metadata["bandwidth_hz"]
    )

    source_bandwidth = (
        1.0 / source_dwell_time
    )

    if not np.isclose(
        source_bandwidth,
        source_bandwidth_stored,
        rtol=1e-6,
        atol=1e-9,
    ):
        raise ValueError(
            "Stored bandwidth and dwell time "
            "are inconsistent:\n"
            f"  bandwidth metadata: "
            f"{source_bandwidth_stored}\n"
            f"  1 / dwell_time: "
            f"{source_bandwidth}"
        )

    source_n_points = int(
        native_fids.shape[-1]
    )

    if (
        source_n_points
        != int(metadata["n_points"])
    ):
        raise ValueError(
            "Stored n_points metadata does "
            "not match the FID arrays."
        )

    hz_per_ppm = float(
        metadata["hz_per_ppm"]
    )

    ppm_reference = (
        _read_common_ppm_reference(
            library_path,
            names,
        )
    )

    source_frequency_resolution = (
        source_bandwidth
        / source_n_points
    )

    resampled_n_points = (
        _choose_resampled_length(
            source_n_points=(
                source_n_points
            ),
            source_frequency_resolution=(
                source_frequency_resolution
            ),
            target_bandwidth=(
                target_bandwidth
            ),
        )
    )

    actual_bandwidth = (
        resampled_n_points
        * source_frequency_resolution
    )

    relative_bandwidth_error = abs(
        actual_bandwidth
        - target_bandwidth
    ) / target_bandwidth

    if (
        relative_bandwidth_error
        > max_relative_bandwidth_error
    ):
        raise ValueError(
            "The requested bandwidth cannot be "
            "represented sufficiently accurately "
            "on the native basis frequency grid.\n"
            f"Requested bandwidth: "
            f"{target_bandwidth:.9f} Hz\n"
            f"Closest bandwidth: "
            f"{actual_bandwidth:.9f} Hz\n"
            f"Relative error: "
            f"{relative_bandwidth_error:.3e}"
        )

    if (
        target_n_timepoints
        > resampled_n_points
    ):
        source_duration = (
            source_n_points
            * source_dwell_time
        )

        target_duration = (
            target_n_timepoints
            / actual_bandwidth
        )

        raise ValueError(
            "The requested output FID is longer "
            "than the resampled native FID.\n"
            f"Native duration: "
            f"{source_duration:.6f} s\n"
            f"Requested duration: "
            f"{target_duration:.6f} s"
        )

    # Transform all metabolites at once.
    native_spectra = np.fft.fftshift(
        np.fft.fft(
            native_fids,
            axis=-1,
        ),
        axes=-1,
    )

    resized_spectra = (
        _resize_centered_spectra(
            native_spectra,
            target_n_points=(
                resampled_n_points
            ),
        )
    )

    # NumPy's inverse FFT includes a 1/N factor. When the FFT length
    # changes, this scaling preserves the original FID amplitude.
    amplitude_scale = (
        resampled_n_points
        / source_n_points
    )

    resized_spectra = (
        resized_spectra
        * amplitude_scale
    )

    resampled_fids = np.fft.ifft(
        np.fft.ifftshift(
            resized_spectra,
            axes=-1,
        ),
        axis=-1,
    )

    prepared_fids = resampled_fids[
    ...,
    :target_n_timepoints,
    ]

    # Match the frequency orientation of the in-vivo and
    # measured water/lipid FIDs used by WALINET.
    prepared_fids = np.conj(
        prepared_fids
    )

    prepared_fids = np.ascontiguousarray(
        prepared_fids,
        dtype=output_dtype,
    )

    actual_dwell_time = (
        1.0 / actual_bandwidth
    )

    result = PreparedBasis(
        names=list(names),
        fids=prepared_fids,
        bandwidth=float(
            actual_bandwidth
        ),
        dwell_time=float(
            actual_dwell_time
        ),
        n_timepoints=int(
            target_n_timepoints
        ),
        hz_per_ppm=hz_per_ppm,
        ppm_reference=ppm_reference,
        requested_bandwidth=float(
            target_bandwidth
        ),
        source_bandwidth=float(
            source_bandwidth
        ),
        source_dwell_time=float(
            source_dwell_time
        ),
        source_n_points=(
            source_n_points
        ),
        dataset_name=dataset_name,
        library_path=library_path,
    )

    if verbose:
        print(
            "Prepared LCModel basis"
        )
        print(
            f"  Library          : "
            f"{library_path}"
        )
        print(
            f"  Dataset          : "
            f"{dataset_name}"
        )
        print(
            f"  Metabolites      : "
            f"{result.n_metabolites}"
        )
        print(
            f"  Output shape     : "
            f"{result.fids.shape}"
        )
        print(
            f"  Source bandwidth : "
            f"{source_bandwidth:.9f} Hz"
        )
        print(
            f"  Target bandwidth : "
            f"{target_bandwidth:.9f} Hz"
        )
        print(
            f"  Actual bandwidth : "
            f"{actual_bandwidth:.9f} Hz"
        )
        print(
            f"  Relative error   : "
            f"{relative_bandwidth_error:.3e}"
        )
        print(
            f"  Dwell time       : "
            f"{actual_dwell_time:.9e} s"
        )
        print(
            f"  Hz / ppm         : "
            f"{hz_per_ppm:.6f}"
        )
        print(
            f"  ppm reference    : "
            f"{ppm_reference:.6f}"
        )
        print(
            f"  Array dtype      : "
            f"{result.fids.dtype}"
        )
        print(
            f"  C contiguous     : "
            f"{result.fids.flags.c_contiguous}"
        )

    return result