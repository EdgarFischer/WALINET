from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_FLOAT_PATTERN = (
    r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[EeDd][-+]?\d+)?"
)
_FLOAT_RE = re.compile(_FLOAT_PATTERN)


@dataclass
class LCModelBasis:
    """
    Parsed LCModel BASIS file.

    Notes
    -----
    `raw_spectra` contains the complex data exactly as stored in the
    LCModel BASIS file. These arrays are not yet converted into the
    final time-domain basis FIDs used for simulation.
    """

    names: list[str]
    raw_spectra: np.ndarray

    dwell_time: float
    hz_per_ppm: float
    echo_time: float
    sequence: str

    ids: list[str]
    concentrations: np.ndarray
    tramps: np.ndarray
    volumes: np.ndarray
    ishifts: np.ndarray

    @property
    def n_metabolites(self) -> int:
        return self.raw_spectra.shape[0]

    @property
    def n_points(self) -> int:
        return self.raw_spectra.shape[1]

    @property
    def sampling_rate(self) -> float:
        return 1.0 / self.dwell_time

    @property
    def bandwidth(self) -> float:
        return self.sampling_rate

    @property
    def time_axis(self) -> np.ndarray:
        return np.arange(self.n_points) * self.dwell_time

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            name: self.raw_spectra[i]
            for i, name in enumerate(self.names)
        }


def _fortran_float(value: str) -> float:
    """Convert Fortran-style D/E floating point text to Python float."""
    return float(
        value.replace("D", "E").replace("d", "e")
    )


def _read_float(block: str, key: str) -> float:
    match = re.search(
        rf"\b{re.escape(key)}\s*=\s*({_FLOAT_PATTERN})",
        block,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not find numeric field '{key}'."
        )

    return _fortran_float(match.group(1))


def _read_int(block: str, key: str) -> int:
    return int(round(_read_float(block, key)))


def _read_string(block: str, key: str) -> str:
    match = re.search(
        rf"\b{re.escape(key)}\s*=\s*'([^']*)'",
        block,
        flags=re.IGNORECASE,
    )

    if match is None:
        raise ValueError(
            f"Could not find string field '{key}'."
        )

    return match.group(1).strip()


def _find_header_block(
    text: str,
    block_name: str,
) -> str:
    match = re.search(
        rf"(?ms)^\s*\${re.escape(block_name)}\s*$"
        rf"\s*(.*?)"
        rf"^\s*\$END\s*$",
        text,
    )

    if match is None:
        raise ValueError(
            f"Could not find ${block_name} block."
        )

    return match.group(1)


def load_lcmodel_basis(
    path: str | Path,
    *,
    dtype: np.dtype = np.complex64,
) -> LCModelBasis:
    """
    Read an LCModel .basis file.

    Parameters
    ----------
    path:
        Path to the LCModel BASIS file.

    dtype:
        Complex dtype used for the stored raw spectra.

    Returns
    -------
    LCModelBasis
        Parsed raw basis spectra and associated metadata.
    """
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Basis file does not exist: {path}"
        )

    text = path.read_text(
        encoding="latin-1",
        errors="strict",
    )

    seqpar_block = _find_header_block(
        text,
        "SEQPAR",
    )

    basis1_block = _find_header_block(
        text,
        "BASIS1",
    )

    hz_per_ppm = _read_float(
        seqpar_block,
        "HZPPPM",
    )

    echo_time = _read_float(
        seqpar_block,
        "ECHOT",
    )

    sequence = _read_string(
        seqpar_block,
        "SEQ",
    )

    dwell_time = _read_float(
        basis1_block,
        "BADELT",
    )

    n_points = _read_int(
        basis1_block,
        "NDATAB",
    )

    basis_block_pattern = re.compile(
        r"(?ms)"
        r"^\s*\$BASIS\s*$"
        r"\s*(.*?)"
        r"^\s*\$END\s*$"
    )

    basis_matches = list(
        basis_block_pattern.finditer(text)
    )

    if not basis_matches:
        raise ValueError(
            "No $BASIS metabolite blocks found."
        )

    names: list[str] = []
    ids: list[str] = []

    concentrations: list[float] = []
    tramps: list[float] = []
    volumes: list[float] = []
    ishifts: list[int] = []

    raw_spectra: list[np.ndarray] = []

    for i, match in enumerate(basis_matches):
        metadata_block = match.group(1)

        data_start = match.end()

        data_end = (
            basis_matches[i + 1].start()
            if i + 1 < len(basis_matches)
            else len(text)
        )

        data_block = text[data_start:data_end]

        next_nmused = re.search(
            r"(?m)^\s*\$NMUSED\s*$",
            data_block,
        )

        if next_nmused is not None:
            data_block = data_block[
                :next_nmused.start()
            ]

        numeric_strings = _FLOAT_RE.findall(
            data_block
        )

        values = np.fromiter(
            (
                _fortran_float(value)
                for value in numeric_strings
            ),
            dtype=np.float64,
        )

        expected_values = 2 * n_points

        if values.size != expected_values:
            metabolite_name = _read_string(
                metadata_block,
                "METABO",
            )

            raise ValueError(
                f"Unexpected number of values for "
                f"'{metabolite_name}': "
                f"found {values.size}, "
                f"expected {expected_values} "
                f"({n_points} complex points)."
            )

        raw_spectrum = (
            values[0::2]
            + 1j * values[1::2]
        )

        names.append(
            _read_string(
                metadata_block,
                "METABO",
            )
        )

        ids.append(
            _read_string(
                metadata_block,
                "ID",
            )
        )

        concentrations.append(
            _read_float(
                metadata_block,
                "CONC",
            )
        )

        tramps.append(
            _read_float(
                metadata_block,
                "TRAMP",
            )
        )

        volumes.append(
            _read_float(
                metadata_block,
                "VOLUME",
            )
        )

        ishifts.append(
            _read_int(
                metadata_block,
                "ISHIFT",
            )
        )

        raw_spectra.append(
            raw_spectrum.astype(
                dtype,
                copy=False,
            )
        )

    raw_spectra_array = np.stack(
        raw_spectra,
        axis=0,
    )

    return LCModelBasis(
        names=names,
        raw_spectra=raw_spectra_array,
        dwell_time=dwell_time,
        hz_per_ppm=hz_per_ppm,
        echo_time=echo_time,
        sequence=sequence,
        ids=ids,
        concentrations=np.asarray(
            concentrations,
            dtype=np.float32,
        ),
        tramps=np.asarray(
            tramps,
            dtype=np.float32,
        ),
        volumes=np.asarray(
            volumes,
            dtype=np.float32,
        ),
        ishifts=np.asarray(
            ishifts,
            dtype=np.int32,
        ),
    )


import numpy as np


def apply_low_ppm_artifact_corrections(
    processed_basis,
    basis,
    corrections: dict[str, dict[str, float]],
):
    """
    Apply metabolite-specific smooth low-ppm suppression to the
    HLSVD-cleaned basis components.

    The corrected clean FIDs are written back into processed_basis.
    The removed-reference FIDs are then recomputed so that

        original_fid
        =
        clean_fid
        +
        reference_fid

    remains exactly satisfied.
    """
    if not corrections:
        print(
            "No low-ppm artifact corrections requested."
        )
        return processed_basis

    names = list(
        processed_basis.names
    )

    missing_metabolites = sorted(
        set(corrections)
        - set(names)
    )

    if missing_metabolites:
        raise KeyError(
            "Correction requested for missing metabolites:\n"
            f"  {missing_metabolites}"
        )

    clean_fids = np.asarray(
        processed_basis.clean_fids
    ).copy()

    original_fids = np.asarray(
        processed_basis.original_fids
    )

    if clean_fids.shape != original_fids.shape:
        raise ValueError(
            "clean_fids and original_fids have "
            "different shapes."
        )

    n_points = int(
        clean_fids.shape[-1]
    )

    dwell_time = float(
        basis.dwell_time
    )

    hz_per_ppm = float(
        basis.hz_per_ppm
    )

    ppm_reference = float(
        processed_basis.ppm_reference
    )

    frequency_axis_hz = np.fft.fftshift(
        np.fft.fftfreq(
            n_points,
            d=dwell_time,
        )
    )

    ppm_axis = (
        ppm_reference
        + frequency_axis_hz / hz_per_ppm
    )

    for metabolite, settings in corrections.items():
        zero_below_ppm = float(
            settings["zero_below_ppm"]
        )

        keep_above_ppm = float(
            settings["keep_above_ppm"]
        )

        if not np.isfinite(
            zero_below_ppm
        ):
            raise ValueError(
                f"{metabolite}: zero_below_ppm "
                "must be finite."
            )

        if not np.isfinite(
            keep_above_ppm
        ):
            raise ValueError(
                f"{metabolite}: keep_above_ppm "
                "must be finite."
            )

        if keep_above_ppm <= zero_below_ppm:
            raise ValueError(
                f"{metabolite}: keep_above_ppm must "
                "be greater than zero_below_ppm."
            )

        metabolite_index = names.index(
            metabolite
        )

        clean_fid = clean_fids[
            metabolite_index
        ]

        clean_spectrum = np.fft.fftshift(
            np.fft.fft(
                clean_fid
            )
        )

        taper = np.ones(
            n_points,
            dtype=np.float64,
        )

        taper[
            ppm_axis <= zero_below_ppm
        ] = 0.0

        transition_mask = (
            (ppm_axis > zero_below_ppm)
            & (ppm_axis < keep_above_ppm)
        )

        transition_position = (
            (
                ppm_axis[transition_mask]
                - zero_below_ppm
            )
            / (
                keep_above_ppm
                - zero_below_ppm
            )
        )

        taper[transition_mask] = (
            0.5
            - 0.5
            * np.cos(
                np.pi
                * transition_position
            )
        )

        corrected_spectrum = (
            clean_spectrum
            * taper
        )

        corrected_fid = np.fft.ifft(
            np.fft.ifftshift(
                corrected_spectrum
            )
        )

        clean_fids[
            metabolite_index
        ] = corrected_fid

        print(
            f"[Low-ppm correction] {metabolite}: "
            f"zero below {zero_below_ppm:.3f} ppm, "
            f"unchanged above {keep_above_ppm:.3f} ppm"
        )

    # Preserve exact decomposition:
    #
    # original = clean + removed reference
    reference_fids = (
        original_fids
        - clean_fids
    )

    processed_basis.clean_fids[...] = (
        clean_fids
    )

    processed_basis.reference_fids[...] = (
        reference_fids
    )

    reconstruction_error = np.max(
        np.abs(
            processed_basis.clean_fids
            + processed_basis.reference_fids
            - processed_basis.original_fids
        )
    )

    print(
        "Maximum reconstruction error after correction:",
        f"{reconstruction_error:.3e}",
    )

    return processed_basis