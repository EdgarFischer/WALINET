from pathlib import Path
import os

import h5py
import numpy as np
import nibabel as nib
from nibabel.processing import resample_from_to
import matplotlib.pyplot as plt


def load_magnitude_downsampled(maps_folder):
    maps_folder = Path(maps_folder)

    mag_nii = nib.load(maps_folder / "magnitude.nii.gz")
    mask_nii = nib.load(maps_folder / "mask.nii.gz")

    mag_resampled_nii = resample_from_to(mag_nii, mask_nii, order=1)
    mag_down = mag_resampled_nii.get_fdata(dtype=np.float32)

    mag_down = mag_down[::-1, ::-1, :]
    mag_down = np.swapaxes(mag_down, 0, 1)

    return mag_down.astype(np.float32)


def prepare_subject_data(base):
    base = Path(base)
    original_data = base / "OriginalData"
    maps = original_data / "maps"
    masks = base / "masks"

    os.makedirs(original_data, exist_ok=True)
    os.makedirs(masks, exist_ok=True)

    with h5py.File(original_data / "CombinedCSI.mat", "r") as f:
        raw = f["csi"]["Data"][:]
        data = raw["real"] + 1j * raw["imag"]
        mask = f["mask"][:]

    data_tr = np.transpose(data, (2, 3, 1, 0))
    mask_tr = np.transpose(mask, (1, 2, 0))

    lipid_mask = nib.load(maps / "mask_lipid.mnc").get_fdata()
    lipid_mask = np.transpose(lipid_mask, (1, 2, 0))

    magnitude = load_magnitude_downsampled(maps)

    np.save(original_data / "data.npy", data_tr)
    np.save(original_data / "magnitude.npy", magnitude)
    np.save(masks / "brain_mask.npy", mask_tr)
    np.save(masks / "lipid_mask.npy", lipid_mask)

    return data_tr, mask_tr, lipid_mask, magnitude


def save_mask_verification_plot(
    data_tr,
    brain_mask,
    lipid_mask,
    base,
    z=15,
    t=4,
    filename="MaskVerification.png",
):
    base = Path(base)
    out_path = base / filename

    lipid_slice = lipid_mask[:, :, z]
    brain_slice = brain_mask[:, :, z]
    data_slice = np.abs(data_tr[:, :, z, t])

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(lipid_slice, cmap="plasma")
    plt.title("Lipid Mask")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(brain_slice, cmap="plasma")
    plt.title("Brain Mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(data_slice, cmap="plasma")
    plt.title(f"Data abs z={z}, t={t}")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    return out_path


def process_subject(base, z=15, t=4):
    data_tr, brain_mask, lipid_mask, magnitude = prepare_subject_data(base)

    verification_path = save_mask_verification_plot(
        data_tr=data_tr,
        brain_mask=brain_mask,
        lipid_mask=lipid_mask,
        base=base,
        z=z,
        t=t,
    )

    return verification_path


def process_subjects(bases, z=15, t=4):
    results = {}

    for base in bases:
        base = Path(base)
        print(f"Processing {base} ...")

        try:
            verification_path = process_subject(base, z=z, t=t)
            results[str(base)] = verification_path
            print(f"Saved {verification_path}")

        except Exception as e:
            results[str(base)] = e
            print(f"Failed {base}: {e}")

    return results