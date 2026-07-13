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