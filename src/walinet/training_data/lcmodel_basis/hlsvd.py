from __future__ import annotations

from dataclasses import dataclass

import hlsvdpropy
import numpy as np

from .parser import LCModelBasis
from .processing import lcmodel_component_to_fid

def model_reference_peak_hlsvd(
    fid: np.ndarray,
    *,
    dwell_time: float,
    hz_per_ppm: float,
    ppm_limits: tuple[float, float] = (-0.2, 0.2),
    ppm_reference: float = 4.65,
    n_singular_values: int = 5,
    n_fit_points: int = 8192,
) -> tuple[np.ndarray, dict]:
    """
    Estimate the reference peak from an initial segment of the FID,
    then reconstruct it over the complete FID.
    """
    fid = np.asarray(fid, dtype=np.complex128)

    if n_fit_points > fid.size:
        raise ValueError(
            f"n_fit_points={n_fit_points} exceeds FID length={fid.size}."
        )

    # Only use a manageable initial segment for the expensive HLSVD fit
    fit_fid = fid[:n_fit_points]
    m = fit_fid.size // 2

    raw_result = hlsvdpropy.hlsvdpro(
        fit_fid,
        n_singular_values,
        m=m,
        sparse=True,
    )

    converted_result = hlsvdpropy.convert_hlsvd_result(
        raw_result,
        dwell_time,
    )

    (
        n_singular_values_found,
        singular_values,
        frequencies_hz,
        damping_times,
        amplitudes,
        phases_deg,
    ) = converted_result[:6]

    frequencies_hz = np.asarray(frequencies_hz)
    damping_times = np.asarray(damping_times)
    amplitudes = np.asarray(amplitudes)
    phases_deg = np.asarray(phases_deg)
    singular_values = np.asarray(singular_values)

    frequency_limits_hz = (
        np.asarray(ppm_limits) - ppm_reference
    ) * hz_per_ppm

    selected = (
        (frequencies_hz > frequency_limits_hz[0])
        & (frequencies_hz < frequency_limits_hz[1])
    )

    # Reconstruct the selected components over the FULL original FID
    full_time_axis = (
        np.arange(fid.size, dtype=np.float64)
        * dwell_time
    )

    reference_fid = np.zeros_like(
        fid,
        dtype=np.complex128,
    )

    for use, frequency, damping, amplitude, phase in zip(
        selected,
        frequencies_hz,
        damping_times,
        amplitudes,
        phases_deg,
    ):
        if use:
            reference_fid += amplitude * np.exp(
                full_time_axis / damping
                + 1j
                * 2.0
                * np.pi
                * (
                    frequency * full_time_axis
                    + phase / 360.0
                )
            )

    component_ppm = (
        frequencies_hz / hz_per_ppm
        + ppm_reference
    )

    info = {
        "n_fit_points": n_fit_points,
        "hankel_shape": (
            fit_fid.size - m,
            m + 1,
        ),
        "n_singular_values_found": n_singular_values_found,
        "singular_values": singular_values,
        "frequencies_hz": frequencies_hz,
        "component_ppm": component_ppm,
        "damping_times": damping_times,
        "amplitudes": amplitudes,
        "phases_deg": phases_deg,
        "selected": selected,
        "selected_frequencies_hz": frequencies_hz[selected],
        "selected_ppm": component_ppm[selected],
        "selected_damping_times": damping_times[selected],
        "selected_amplitudes": amplitudes[selected],
        "selected_phases_deg": phases_deg[selected],
    }

    return reference_fid, info

@dataclass
class ProcessedLCModelBasis:
    """
    Result of HLSVD reference-peak removal for a complete LCModel basis.
    """

    names: list[str]

    original_fids: np.ndarray
    clean_fids: np.ndarray
    reference_fids: np.ndarray

    hlsvd_info_by_metabolite: dict[str, dict]

    ppm_limits: tuple[float, float]
    ppm_reference: float
    n_singular_values: int
    n_fit_points: int

    @property
    def n_metabolites(self) -> int:
        return len(self.names)

    @property
    def n_points(self) -> int:
        return self.clean_fids.shape[-1]


def process_lcmodel_basis(
    basis: LCModelBasis,
    *,
    ppm_limits: tuple[float, float] = (-0.2, 0.2),
    ppm_reference: float = 4.65,
    n_singular_values: int = 5,
    n_fit_points: int = 8192,
    verbose: bool = True,
) -> ProcessedLCModelBasis:
    """
    Remove the artificial LCModel reference peak from every basis
    component using HLSVD.

    Parameters
    ----------
    basis:
        Parsed LCModel basis.

    ppm_limits:
        ppm interval containing the artificial reference peak.

    ppm_reference:
        ppm reference used for converting HLSVD frequencies.

    n_singular_values:
        Number of HLSVD components to estimate.

    n_fit_points:
        Number of initial native FID points used for the HLSVD fit.

    verbose:
        Print progress and selected reference components.

    Returns
    -------
    ProcessedLCModelBasis
        Original, cleaned and removed-reference FIDs for all components.
    """
    original_fids: list[np.ndarray] = []
    clean_fids: list[np.ndarray] = []
    reference_fids: list[np.ndarray] = []

    hlsvd_info_by_metabolite: dict[str, dict] = {}

    for metabolite in basis.names:
        if verbose:
            print(f"Processing {metabolite}...")

        original_fid = lcmodel_component_to_fid(
            basis=basis,
            metabolite=metabolite,
        )

        reference_fid, hlsvd_info = (
            model_reference_peak_hlsvd(
                original_fid,
                dwell_time=basis.dwell_time,
                hz_per_ppm=basis.hz_per_ppm,
                ppm_limits=ppm_limits,
                ppm_reference=ppm_reference,
                n_singular_values=n_singular_values,
                n_fit_points=n_fit_points,
            )
        )

        clean_fid = original_fid - reference_fid

        original_fids.append(original_fid)
        clean_fids.append(clean_fid)
        reference_fids.append(reference_fid)

        hlsvd_info_by_metabolite[metabolite] = hlsvd_info

        if verbose:
            selected_ppm = hlsvd_info["selected_ppm"]

            print(
                f"  removed {selected_ppm.size} component(s): "
                f"{selected_ppm}"
            )

    original_fids_array = np.stack(
        original_fids,
        axis=0,
    )

    clean_fids_array = np.stack(
        clean_fids,
        axis=0,
    )

    reference_fids_array = np.stack(
        reference_fids,
        axis=0,
    )

    expected_shape = (
        basis.n_metabolites,
        basis.n_points,
    )

    for array_name, array in (
        ("original_fids", original_fids_array),
        ("clean_fids", clean_fids_array),
        ("reference_fids", reference_fids_array),
    ):
        if array.shape != expected_shape:
            raise RuntimeError(
                f"{array_name} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

    reconstruction_error = np.max(
        np.abs(
            clean_fids_array
            + reference_fids_array
            - original_fids_array
        )
    )

    if verbose:
        print()
        print(
            "Maximum reconstruction error:",
            reconstruction_error,
        )

    return ProcessedLCModelBasis(
        names=list(basis.names),
        original_fids=original_fids_array,
        clean_fids=clean_fids_array,
        reference_fids=reference_fids_array,
        hlsvd_info_by_metabolite=hlsvd_info_by_metabolite,
        ppm_limits=ppm_limits,
        ppm_reference=ppm_reference,
        n_singular_values=n_singular_values,
        n_fit_points=n_fit_points,
    )