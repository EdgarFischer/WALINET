import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from walinet.inference.fid_inference import infer_fid

infer_fid(
    fid="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/NoB0Correction/MS_180/OriginalData/data.npy",
    headmask="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/NoB0Correction/MS_180/masks/brain_mask.npy",
    model_dir="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/models/7T_Final",
    output_path="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/NoB0Correction/MS_180/OriginalData/data_test.npy",
    fid_axis="auto",
    device="cuda:0",
)