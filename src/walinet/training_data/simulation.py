# src/walinet/training_data/simulation.py

from pathlib import Path
import glob
import re
import h5py

import numpy as np
import pandas as pd
from tqdm import tqdm

def read_mode_files(index: dict, list_file: list[str]) -> list[np.ndarray]:
    n_metabolites = len(list_file)
    modes = [None] * n_metabolites

    for filename in list_file:
        metabo_mode = pd.read_csv(filename, header=None, skiprows=[0]).values

        match = re.search(r"[0-9]T_.{1,6}_Exact", filename)
        if match is None:
            raise ValueError(f"Could not extract metabolite name from: {filename}")

        name = bytes(filename[match.span()[0] + 3 : match.span()[1] - 6].strip(), "utf8")
        modes[index[name]] = metabo_mode

    return modes


def load_metabolite_modes(
    mean_std_path: str | Path,
    modes_glob: str,
):
    mean_std_csv = pd.read_csv(mean_std_path, header=None).values

    index = {}
    for i, value in enumerate(mean_std_csv[:, 0].astype(str)):
        index[bytes(value.strip(), "utf8")] = i

    mean_std = mean_std_csv[:, 1:].astype(np.float32)

    list_file = glob.glob(modes_glob)
    if len(list_file) == 0:
        raise FileNotFoundError(f"No metabolite mode files found for glob: {modes_glob}")

    n_metabolites = len(list_file)
    metabo_modes = [[[None] for _ in range(n_metabolites)] for _ in range(6)]
    metabo_modes[0] = read_mode_files(index, list_file)

    return metabo_modes, mean_std


def simulate_metabolite_spectra(
    *,
    rng: np.random.Generator,
    n_spectra: int,
    n_timepoints: int,
    sampling_rate: float,
    nmr_freq: float,
    mean_std_path: str | Path,
    modes_glob: str,
    max_acqu_delay: float,
    max_freq_shift: float,
    min_peak_width: float,
    max_peak_width: float,
    min_snr: float,
    max_snr: float,
) -> np.ndarray:
    """
    Simulate metabolite spectra.

    Returns:
        metab_spectrum: (n_spectra, n_timepoints), complex
    """
    t = np.arange(n_timepoints) / sampling_rate

    acqu_delay = (rng.random((n_spectra, 1)) - 0.5) * 2 * max_acqu_delay
    phase_shift = rng.random((n_spectra, 1)) * 2 * np.pi
    freq_shift = (rng.random((n_spectra, 1)) * 2 - 1) * max_freq_shift

    peak_width = min_peak_width + rng.random((n_spectra, 1)) * (
        max_peak_width - min_peak_width
    )
    ponder_peaks = rng.random((n_spectra, 1))
    peak_width_gau = ponder_peaks * peak_width
    peak_width_lor = (1 - ponder_peaks) * peak_width

    snr = min_snr + rng.random((n_spectra, 1)) * (max_snr - min_snr)

    metabo_modes, mean_std = load_metabolite_modes(
        mean_std_path=mean_std_path,
        modes_glob=modes_glob,
    )

    n_metabolites = len(metabo_modes[0])

    temp_metab_data = np.zeros(
        (n_metabolites, n_timepoints),
        dtype=np.complex64,
    )
    time_series_clean = np.zeros(
        (n_timepoints,),
        dtype=np.complex64,
    )

    amplitude = mean_std[:, 1] * rng.standard_normal((n_spectra, n_metabolites)) + mean_std[:, 0]
    amplitude = amplitude.clip(min=0)

    metab_spectrum = np.zeros(
        (n_spectra, n_timepoints),
        dtype=np.complex128,
    )

    for n in tqdm(range(n_spectra), miniters=100):
        temp_metab_data[:] = 0

        for f, mode in enumerate(metabo_modes[0]):
            freq = ((4.7 - mode[:, 0]) * 1e-6 * nmr_freq)[..., None]

            for nuc in range(len(freq)):
                if (mode[nuc, 0] > 0.0) & (mode[nuc, 0] < 4.5):
                    temp_metab_data[f, :] += (
                        mode[nuc, 1][..., None]
                        * np.exp(1j * mode[nuc, 2][..., None])
                        * np.exp(
                            2
                            * np.pi
                            * 1j
                            * (t + acqu_delay[n])
                            * freq[nuc]
                        )
                    )

        time_series_clean[:] = 0

        for f, _ in enumerate(metabo_modes[0]):
            time_series_clean[:] += (
                amplitude[n, f]
                * temp_metab_data[f, :]
                * np.exp(1j * phase_shift[n])
            )

        time_series_clean[:] *= np.exp(
            (t * 1j * 2 * np.pi * freq_shift[n])
            - (np.square(t) * np.square(peak_width_gau[n]))
            - (np.abs(t) * peak_width_lor[n])
        )

        spectrum_temp = np.fft.fftshift(np.fft.fft(time_series_clean, axis=0))

        noise = rng.standard_normal(n_timepoints) + 1j * rng.standard_normal(n_timepoints)

        time_series = time_series_clean + np.fft.ifft(
            spectrum_temp.std() / 0.65 / snr[n] * noise,
            axis=0,
        )

        metab_spectrum[n] = np.fft.fftshift(np.fft.fft(time_series))

    return metab_spectrum


def simulate_lipid_spectra(
    *,
    rng: np.random.Generator,
    image_rrrt: np.ndarray,
    lipid_mask: np.ndarray,
    metab_spectrum: np.ndarray,
    n_random_lipid: int,
    max_lipid_scaling: float,
) -> np.ndarray:
    """
    Simulate lipid spectra from real lipid-mask voxels.

    Returns:
        lipid_rf: (n_spectra, n_timepoints)
    """
    n_spectra, n_timepoints = metab_spectrum.shape

    lipid_rf = np.zeros(
        (n_spectra, n_timepoints),
        dtype=np.complex64,
    )

    image_rrrf = np.fft.fftshift(
        np.fft.fft(image_rrrt, axis=-1),
        axes=-1,
    )

    nonzero_indices = np.nonzero(lipid_mask)
    n_lipid_voxels = int(np.sum(lipid_mask))

    for i in range(n_spectra):
        indices = rng.choice(n_lipid_voxels, size=n_random_lipid)

        xx = nonzero_indices[0][indices]
        yy = nonzero_indices[1][indices]
        zz = nonzero_indices[2][indices]

        lipid_batch = image_rrrf[xx, yy, zz, :]

        lip_amp = rng.random(n_random_lipid)
        lip_amp = lip_amp / np.sum(lip_amp)

        lipid_rf[i] = np.sum(
            lipid_batch * lip_amp[:, None],
            axis=0,
        )

    lip_max = np.amax(np.abs(lipid_rf), axis=1)[:, None]
    metab_max = np.amax(np.abs(metab_spectrum), axis=1)[:, None]

    lipid_scaling = 1e-1 * (
        10 ** (rng.random((n_spectra, 1)) * np.log10(1e1 * max_lipid_scaling))
    )

    lipid_rf = metab_max / lip_max * lipid_scaling * lipid_rf

    return lipid_rf


def simulate_water_spectra(
    *,
    rng: np.random.Generator,
    water_rrrt: np.ndarray,
    brain_mask: np.ndarray,
    metab_spectrum: np.ndarray,
    water_scaling_min: float = 0.0,
    water_scaling_max: float = 100.0,
) -> np.ndarray:
    """
    Simulate water spectra from isolated-water voxels.

    Returns:
        water_rf: (n_spectra, n_timepoints)
    """
    n_spectra, n_timepoints = metab_spectrum.shape

    water_rf = np.zeros(
        (n_spectra, n_timepoints),
        dtype=np.complex64,
    )

    water_rrrf = np.fft.fftshift(
        np.fft.fft(water_rrrt, axis=-1),
        axes=-1,
    )

    nonzero_indices = np.nonzero(brain_mask)
    n_brain_voxels = int(np.sum(brain_mask))

    for i in range(n_spectra):
        indices = rng.choice(n_brain_voxels, size=1)

        xx = nonzero_indices[0][indices]
        yy = nonzero_indices[1][indices]
        zz = nonzero_indices[2][indices]

        water_batch = water_rrrf[xx, yy, zz, :]

        water_amp = rng.random(1) + 0.5

        water_rf[i] = np.sum(
            water_batch * water_amp[:, None],
            axis=0,
        )

    water_max = np.amax(np.abs(water_rf), axis=1)[:, None]
    metab_max = np.amax(np.abs(metab_spectrum), axis=1)[:, None]

    water_scaling = rng.uniform(
    water_scaling_min,
    water_scaling_max,
    size=(n_spectra, 1),
)

    water_rf = metab_max / water_max * water_scaling * water_rf

    return water_rf


def assemble_training_spectra(
    *,
    metab_spectrum: np.ndarray,
    lipid_rf: np.ndarray,
    water_rf: np.ndarray,
    lipid_proj_operator_ff: np.ndarray,
):
    spectra = water_rf + lipid_rf + metab_spectrum
    lipid_proj = np.matmul(spectra, lipid_proj_operator_ff)

    return spectra, lipid_proj


def finite_row_mask(*arrays: np.ndarray) -> np.ndarray:
    """
    Return mask for rows where all provided arrays are finite.
    Assumes arrays have first dimension = n_spectra.
    """
    if len(arrays) == 0:
        raise ValueError("At least one array must be provided.")

    keep_mask = np.ones(arrays[0].shape[0], dtype=bool)

    for arr in arrays:
        bad_rows = np.unique(np.argwhere(~np.isfinite(arr))[:, 0])
        keep_mask &= ~np.isin(np.arange(arr.shape[0]), bad_rows)

    return keep_mask

def process_subject(
    *,
    sub: str,
    path: str | Path,
    version: str,
    rng: np.random.Generator,
    n_spectra: int,
    n_random_lipid: int,
    max_lipid_scaling: float,
    min_snr: float,
    max_snr: float,
    n_timepoints: int,
    sampling_rate: float,
    nmr_freq: float,
    max_freq_shift: float,
    min_peak_width: float,
    max_peak_width: float,
    max_acqu_delay: float,
    water_scaling_min: float,
    water_scaling_max: float,
    mean_std_path: str | Path,
    modes_glob: str,
    lipid_projection_target: float,
    lipid_projection_tol: float,
    lipid_projection_max_iter: int,
):
    subject_dir = Path(path) / sub

    train_data_dir = subject_dir / "TrainData"
    h5_path = train_data_dir / f"TrainData_{version}.h5"

    if h5_path.exists():
        raise FileExistsError(
            f"Training data already exists for subject '{sub}':\n"
            f"  {h5_path}\n\n"
            f"This version name is already used. "
            f"Please specify a different data version."
        )

    p_mask = subject_dir / "masks" / "brain_mask.npy"
    p_cc = subject_dir / "OriginalData" / "data.npy"
    p_scalp_mask = subject_dir / "masks" / "lipid_mask.npy"

    brainmask = np.load(p_mask)
    csi_rrrt = np.load(p_cc)
    skmask = np.load(p_scalp_mask)

    water_rrrt = np.load(subject_dir / "TrainData" / "IsolatedWater_v_1.0.npy")
    image_rrrt = csi_rrrt - water_rrrt
    print("loaded isolated water and reconstructed water-suppressed data")

    data_rrrf = np.fft.fftshift(
        np.fft.fft(csi_rrrt, axis=-1),
        axes=-1,
    )

    lipid_proj_operator_ff = compute_lipid_projection_operator(
        spectra=data_rrrf,
        lipid_mask=skmask,
        target=lipid_projection_target,
        tol=lipid_projection_tol,
        max_n_iter=lipid_projection_max_iter,
    )

    metab_spectrum = simulate_metabolite_spectra(
        rng=rng,
        n_spectra=n_spectra,
        n_timepoints=n_timepoints,
        sampling_rate=sampling_rate,
        nmr_freq=nmr_freq,
        mean_std_path=mean_std_path,
        modes_glob=modes_glob,
        max_acqu_delay=max_acqu_delay,
        max_freq_shift=max_freq_shift,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        min_snr=min_snr,
        max_snr=max_snr,
    )

    lipid_rf = simulate_lipid_spectra(
        rng=rng,
        image_rrrt=image_rrrt,
        lipid_mask=skmask,
        metab_spectrum=metab_spectrum,
        n_random_lipid=n_random_lipid,
        max_lipid_scaling=max_lipid_scaling,
    )

    water_rf = simulate_water_spectra(
        rng=rng,
        water_rrrt=water_rrrt,
        brain_mask=brainmask,
        metab_spectrum=metab_spectrum,
        water_scaling_min=water_scaling_min,
        water_scaling_max=water_scaling_max,
    )

    spectra, lipid_proj = assemble_training_spectra(
        metab_spectrum=metab_spectrum,
        lipid_rf=lipid_rf,
        water_rf=water_rf,
        lipid_proj_operator_ff=lipid_proj_operator_ff,
    )

    keep_mask = finite_row_mask(spectra)
    bad_rows = np.where(~keep_mask)[0]

    print(f"[Clean] Gefundene fehlerhafte Zeilen: {bad_rows.tolist()}")
    print(f"[Clean] Behalte {keep_mask.sum()} von {spectra.shape[0]} Zeilen")

    train_data_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "w") as hf:
        hf.create_dataset("metab", data=metab_spectrum[keep_mask])
        hf.create_dataset("water", data=water_rf[keep_mask])
        hf.create_dataset("spectra", data=spectra[keep_mask])
        hf.create_dataset("lipid_proj", data=lipid_proj[keep_mask])
        hf.create_dataset("lipid", data=lipid_rf[keep_mask])
        hf.create_dataset("lipid_projOP", data=lipid_proj_operator_ff)

    print(f"Saved: {h5_path}")
