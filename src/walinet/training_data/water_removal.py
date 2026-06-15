# src/walinet/training_data/water_removal.py

from pathlib import Path
from itertools import product
import time

import numpy as np
import scipy.linalg
from joblib import Parallel, delayed
from tqdm import tqdm

from walinet.config.schema_training_data import TrainingDataConfig, WaterRemovalCfg


def hsvd(y: np.ndarray, fs: float, k: int):
    n = len(y)
    l = int(np.floor(0.5 * n))

    hankel_matrix = scipy.linalg.hankel(y[:l], y[l:n])
    u, _, _ = np.linalg.svd(hankel_matrix)

    uk = u[:, :k]
    uk_top = uk[1:, :]
    uk_bottom = uk[:-1, :]

    z_prime = np.matmul(np.linalg.pinv(uk_bottom), uk_top)

    eigenvalues, eigenvectors = np.linalg.eig(z_prime)
    z = np.matmul(np.matmul(np.linalg.inv(eigenvectors), z_prime), eigenvectors)

    q = np.log(np.diag(z))
    dt = 1 / fs

    dampings = np.real(q) / dt
    dampings[dampings > 10] = 10

    frequencies = np.imag(q) / (2 * np.pi) / dt

    t = np.arange(start=0, stop=len(y) * dt, step=dt)
    basis = np.exp(
        np.matmul(
            t[:, None],
            (dampings + 2 * np.pi * 1j * frequencies)[None, :],
        )
    )

    amplitudes = np.matmul(np.linalg.pinv(basis), y)

    return frequencies, dampings, basis, amplitudes


def water_suppression_wrapper(
    image_rrrt: np.ndarray,
    mask: np.ndarray,
    fs: float,
    k: int,
    min_freq: float,
    max_freq: float,
):
    def water_suppression(tup):
        x, y, z = tup
        fid = image_rrrt[x, y, z, :]

        if mask[x, y, z]:
            frequencies, _, basis, amplitudes = hsvd(
                y=fid,
                fs=fs,
                k=k,
            )

            idx = np.where(
                np.logical_and(
                    frequencies >= min_freq,
                    frequencies <= max_freq,
                )
            )[0]

            water_fid = np.sum(
                np.matmul(basis[:, idx], np.diag(amplitudes[idx])),
                axis=1,
            )

            return x, y, z, water_fid

        return None

    return water_suppression


def get_subject_paths(cfg: TrainingDataConfig, subject: str) -> dict:
    subject_path = Path(cfg.data.base_dir) / subject

    return {
        "brain_mask": subject_path / cfg.data.paths.brain_mask,
        "data": subject_path / cfg.data.paths.input_data,
        "lipid_mask": subject_path / cfg.data.paths.lipid_mask,
        "output_dir": subject_path / cfg.data.paths.output_dir,
    }


def get_isolated_water_path(cfg: TrainingDataConfig, subject: str) -> Path:
    paths = get_subject_paths(cfg, subject)

    filename = cfg.output.isolated_water_filename.format(
        version=cfg.output.version
    )

    return paths["output_dir"] / filename


def load_subject_data(paths: dict):
    brain_mask = np.load(paths["brain_mask"])
    csi_rrrt = np.load(paths["data"])
    lipid_mask = np.load(paths["lipid_mask"])

    return brain_mask, csi_rrrt, lipid_mask


def make_slice_batches(n_slices: int, batch_size: int):
    slices = []
    current_batch = []

    for i in range(n_slices):
        if len(current_batch) < batch_size - 1 and i < n_slices - 1:
            current_batch.append(i)
        else:
            current_batch.append(i)
            slices.append(tuple(current_batch))
            current_batch = []

    return slices


def suppress_water_volume(
    image_grid: np.ndarray,
    mask: np.ndarray,
    cfg: WaterRemovalCfg,
) -> np.ndarray:
    shape = image_grid.shape

    water_rrrt = np.zeros(image_grid.shape, dtype=np.complex64)

    water_suppression = water_suppression_wrapper(
        image_rrrt=image_grid,
        mask=mask,
        fs=1 / cfg.dwell_time,
        k=cfg.hsvd_components,
        min_freq=cfg.min_freq,
        max_freq=cfg.max_freq,
    )

    slices = make_slice_batches(
        n_slices=shape[2],
        batch_size=cfg.slice_batch_size,
    )

    print("All slices:", slices)

    for sl in slices:
        print("Slice:", sl)

        results = Parallel(n_jobs=cfg.parallel_jobs)(
            delayed(water_suppression)(tup=tup)
            for tup in tqdm(
                product(range(shape[0]), range(shape[1]), sl),
                total=shape[0] * shape[1] * len(sl),
                position=0,
                leave=True,
            )
        )

        for result in results:
            if result is not None:
                x, y, z, water_fid = result
                water_rrrt[x, y, z] = water_fid

    return water_rrrt


def compute_isolated_water(
    subject: str,
    cfg: TrainingDataConfig,
) -> np.ndarray:
    print(f"####### Water suppression: {subject} #######")
    start_time = time.time()

    paths = get_subject_paths(cfg, subject)

    brain_mask, csi_rrrt, lipid_mask = load_subject_data(paths)
    head_mask = brain_mask + lipid_mask

    image_grid = np.array(csi_rrrt)

    water_rrrt = suppress_water_volume(
        image_grid=image_grid,
        mask=head_mask,
        cfg=cfg.water_removal,
    )

    elapsed = time.time() - start_time
    print(f"Water removal finished for {subject}: {elapsed:.2f} s")

    return water_rrrt


def save_isolated_water(
    water_rrrt: np.ndarray,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, water_rrrt)


def get_or_create_isolated_water(
    subject: str,
    cfg: TrainingDataConfig,
) -> np.ndarray:
    output_path = get_isolated_water_path(cfg, subject)

    if output_path.exists() and not cfg.output.overwrite:
        print(f"[Water] Found existing file, loading: {output_path}")
        return np.load(output_path)

    if output_path.exists() and cfg.output.overwrite:
        print(f"[Water] Existing file will be overwritten: {output_path}")

    if not cfg.water_removal.enabled:
        raise FileNotFoundError(
            f"Water removal is disabled and no cached isolated water file "
            f"was found for subject '{subject}': {output_path}"
        )

    print(f"[Water] Computing isolated water: {subject}")
    water_rrrt = compute_isolated_water(subject, cfg)

    print(f"[Water] Saving isolated water: {output_path}")
    save_isolated_water(water_rrrt, output_path)

    return water_rrrt