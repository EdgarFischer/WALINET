import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("../src"))
sys.path.append(os.path.abspath(".."))
from walinet.data.dataprep import * 

Resolution="50x50"

subjects = [
    f"Vol01_WB/Res{Resolution}",
    f"Vol02_BS/Res{Resolution}",
    f"Vol03_SH/Res{Resolution}",
    f"Vol04_SM/Res{Resolution}",
    f"Vol05_LH/Res{Resolution}",
]

bases = [Path(f"/workspace/walinet/data/3T/{s}") for s in subjects]

process_subjects(bases, z=15, t=4)