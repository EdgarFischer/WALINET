import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from walinet.inference.ford_pipeline import run_walinet_ford_pipeline


base = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/hfischer/ProtonFits/7T"
)

goal = (
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/hfischer/ProtonFits/7T/Lukas"
)

# data_base = (
#     "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/"
#     "bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/"
#     "LargeData_d3hj/MeasAndLogData"
# )

subjects = [
    "maps_new_pipeline",
]

# siemens_data = [
#     "UCSF/Volunteer4_20250912/"
#     "meas_MID00100_FID33703_csi_fid_ViennaCrt_v1_01.dat",

#     "London/Vol1_20250612_M701118_METAHEAD/"
#     "meas_MID00109_FID32131_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",

#     "London/Vol5_20250626_M701130_METAHEAD/"
#     "meas_MID00099_FID33074_csi_fid_ViennaCrt_v1a_released_3_4iso.dat",

#     "Vienna/Vol8_AnnaZ/"
#     "meas_MID00166_FID35660_csi_fid_ViennaCrt_v1_00.dat",
# ]


for sub in subjects:
    print(f"\nProcessing {sub}")

    run_walinet_ford_pipeline(
        data_path="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/NoB0Correction/Lukas/OriginalData/data.npy",  #f"{base}/{sub}/CombinedCSI.mat",
        mask_path="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/Denoising/datasets/Proton/7T/NoB0Correction/Lukas/masks/brain_mask.npy",

        walinet_model_dir=(
            "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/"
            "hfish/walinet/models/7T_Final"
        ),
        ford_config_template=(
            "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/"
            "hfish/forD/runs/classical_fitting_config_trained Hauke.json"
        ),
        output_path=f"{goal}/{sub}",
        gpu_number=2,

        fid_axis="auto",
        walinet_checkpoint="model_best.pt",
        walinet_batch_size=200,

        b0_correction=False,
 #       dat_path=f"{data_base}/{dat_file}",

        julia_executable=(
            "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/"
            "hfish/walinet/B0_correction/julia-1.11.1/bin/julia"
        ),
        julia_project=(
            "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/"
            "hfish/walinet/B0_correction"
        ),

        shm_dir="/dev/shm",
    )