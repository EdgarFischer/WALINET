import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))
from walinet.inference.fid_inference import infer_combined_csi

infer_combined_csi(
    input_path="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/hfischer/ProtonFits/7T/Lukas/CombinedCSI.mat",
    model_dir="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/models/7T_Final",
    output_path="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/hfischer/ProtonFits/7T/Lukas/CombinedCSI_NEW_WALINET.mat",
    device="cuda:0",
    # b0_correction=True,
    # dat_path="/path/to/raw_measurement.dat",  # Optional override for Par.Paths.csi_path
    # julia_executable="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/B0_correction/julia-1.11.1/bin/julia",
    # julia_project="/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet/B0_correction",
    # shm_dir="/dev/shm",
)

# London
#Vol01: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol1_20250612_M701118_METAHEAD/meas_MID00109_FID32131_csi_fid_ViennaCrt_v1a_released_3_4iso.dat
#Vol02: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol2_20250616_M701121_METAHEAD/meas_MID00087_FID32273_csi_fid_ViennaCrt_v1a_released_3_4iso.dat
#Vol03: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol3_20250620_M701126_METAHEAD/meas_MID00086_FID32881_csi_fid_ViennaCrt_v1a_released_3_4iso.dat
#Vol04: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol4_20250620_M701128_METAHEAD/meas_MID00162_FID32957_csi_fid_ViennaCrt_v1a_released_3_4iso.dat
#Vol05: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/London/Vol5_20250626_M701130_METAHEAD/meas_MID00099_FID33074_csi_fid_ViennaCrt_v1a_released_3_4iso.dat

# Vienna

#Vol05: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol5_Berni/meas_MID00025_FID27701_csi_fidesi_crt_Feb2025_2.dat
#Vol06: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol6_LukasH/meas_MID00093_FID30255_csi_fidesi_crt_Feb2025_2.dat
#Vol07: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol7_WolfgangB/meas_MID00044_FID35273_csi_fidesi_crt_Feb2025_2.dat
#Vol08: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol8_AnnaZ/meas_MID00166_FID35660_csi_fid_ViennaCrt_v1_00.dat
#Vol09: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Vienna/Vol9_StanoM/meas_MID00054_FID38080_csi_fidesi_crt_Feb2025_2.dat

# UCSF
#Vol01: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer1_20250819/meas_MID00229_FID32904_csi_fid_ViennaCrt_v1_01.dat
#Vol02: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer2_20250827/meas_MID00032_FID33306_csi_fid_ViennaCrt_v1_01.dat
#Vol03: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer3_20250827/meas_MID00068_FID33342_csi_fid_ViennaCrt_v1_01.dat
#Vol04: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer4_20250912/meas_MID00100_FID33703_csi_fid_ViennaCrt_v1_01.dat
#Vol05: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/UCSF/Volunteer5_20250912/meas_MID00033_FID33731_csi_fid_ViennaCrt_v1_01.dat

# Brisbane

#Vol02: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-2/meas_MID00166_FID04729_csi_fid_ViennaCrt_v1_01.dat
#Vol03: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-3/meas_MID00035_FID06136_csi_fid_ViennaCrt_v1_01.dat
#Vol04: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-4/meas_MID00039_FID06170_csi_fid_ViennaCrt_v1_01.dat
#Vol05: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-5/meas_MID00177_FID07765_csi_fid_ViennaCrt_v1_01.dat
#Vol07: /ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/home/bstrasser/Projects/Project9_ImplementRecoInICE/Step5_MultiCenterStudy/LargeData_d3hj/MeasAndLogData/Brisbane/MRSI-TEST-7/meas_MID00033_FID13892_csi_fid_ViennaCrt_v1_01.dat