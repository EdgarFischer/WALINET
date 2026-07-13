from __future__ import annotations

import numpy as np

from .parser import LCModelBasis


def prepare_basis_fid_for_acquisition(
    native_spectrum: np.ndarray,
    *,
    source_dwell_time: float,
    target_bandwidth: float,
    target_n_timepoints: int,
):
    """
    Convert a native LCModel basis spectrum to the desired acquisition.

    The spectrum is cropped to the requested bandwidth and transformed
    into the time domain. Finally, the desired number of FID points is
    retained.

    Parameters
    ----------
    native_spectrum:
        Native LCModel basis spectrum in fftshift ordering.

    source_dwell_time:
        Native LCModel dwell time (BADELT).

    target_bandwidth:
        Desired acquisition bandwidth in Hz.

    target_n_timepoints:
        Desired number of acquired FID points.

    Returns
    -------
    target_fid:
        Cropped time-domain basis function.

    actual_bandwidth:
        Actual bandwidth after integer frequency cropping.
    """
    n_source = native_spectrum.size

    source_frequency_resolution = (
        1.0
        / (n_source * source_dwell_time)
    )

    n_cropped = int(
        round(
            target_bandwidth
            / source_frequency_resolution
        )
    )

    if n_cropped > n_source:
        raise ValueError(
            "Target bandwidth exceeds native basis bandwidth."
        )

    center = n_source // 2

    start = center - n_cropped // 2
    stop = start + n_cropped

    cropped_spectrum = native_spectrum[start:stop]

    actual_bandwidth = (
        n_cropped
        * source_frequency_resolution
    )

    full_fid = np.fft.ifft(
        np.fft.ifftshift(cropped_spectrum)
    )

    if target_n_timepoints > full_fid.size:
        raise ValueError(
            f"Requested {target_n_timepoints} time points, "
            f"but only {full_fid.size} are available."
        )

    target_fid = full_fid[:target_n_timepoints]

    return target_fid, actual_bandwidth

def lcmodel_component_to_fid(
    basis: LCModelBasis,
    metabolite: str,
) -> np.ndarray:
    """
    Convert one raw LCModel BASIS component to a time-domain FID.

    Processing steps
    ----------------
    1. Select the requested metabolite.
    2. Apply the LCModel ISHIFT value.
    3. Transform to the time domain using an inverse FFT.
    4. Apply complex conjugation.
    """
    metabolite_index = basis.names.index(metabolite)

    raw_spectrum = np.asarray(
        basis.raw_spectra[metabolite_index],
        dtype=np.complex128,
    )

    ishift = int(
        basis.ishifts[metabolite_index]
    )

    shifted_spectrum = np.roll(
        raw_spectrum,
        -ishift,
    )

    fid = np.conj(
        np.fft.ifft(shifted_spectrum)
    )

    return fid