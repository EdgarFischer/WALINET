import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

os.chdir(ROOT)

from walinet.training_data.simulation import process_subject

# Notes:
# v1 Test
# v2 Test
# v3 First Dataset

version = "v_2.0"
path = "data/3T/"

subjects = [
    "Vol01_WB/Res64x64_Thick"
]

# ----------------
# Acquisition
# ----------------

bandwidth = 939.85  # Vienna: 7T: 2778; 3T: 939.85
dwell_time = 1 / bandwidth
sampling_rate = 1 / dwell_time

n_timepoints = 288
nmr_freq = 123231706.0  # Vienna: 7T: 297222931, 3T: 123231706

# ----------------
# Simulation
# ----------------

n_spectra = 10
n_random_lipid = 10
max_lipid_scaling = 70

min_snr = 1
max_snr = 10

max_freq_shift = 40
min_peak_width = 20
max_peak_width = 100
max_acqu_delay = 0.002

water_scaling_min = 0.0
water_scaling_max = 100.0

mean_std_path = "MetabModes/Metab_Mean_STD.txt"
modes_glob = "MetabModes/3T_TE0/*Exact_Modes.txt"

# ----------------
# Lipid projection
# ----------------

lipid_projection_target = 0.938
lipid_projection_tol = 5e-3
lipid_projection_max_iter = 60

# ----------------
# Reproducibility
# ----------------

np.random.seed(42)
rng = np.random

for sub in subjects:
    process_subject(
        sub=sub,
        path=path,
        version=version,
        rng=rng,
        n_spectra=n_spectra,
        n_random_lipid=n_random_lipid,
        max_lipid_scaling=max_lipid_scaling,
        min_snr=min_snr,
        max_snr=max_snr,
        n_timepoints=n_timepoints,
        sampling_rate=sampling_rate,
        nmr_freq=nmr_freq,
        max_freq_shift=max_freq_shift,
        min_peak_width=min_peak_width,
        max_peak_width=max_peak_width,
        max_acqu_delay=max_acqu_delay,
        water_scaling_min=water_scaling_min,
        water_scaling_max=water_scaling_max,
        mean_std_path=mean_std_path,
        modes_glob=modes_glob,
        lipid_projection_target=lipid_projection_target,
        lipid_projection_tol=lipid_projection_tol,
        lipid_projection_max_iter=lipid_projection_max_iter,
    )

print("All done!")