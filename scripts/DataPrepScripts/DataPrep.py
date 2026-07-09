import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath(".."))
from walinet.data.dataprep import * 


subjects = [
    #"Vol01_PW",
    #"Vol02_BS",
    #"Vol03_TE",
    #"Vol04_AA",
    "Vol05_LH",
]

Res = [
    "Res36x36",
    "Res50x50",
    "Res64x64x41",
    #"Res64x64x47"
]

for v in subjects:

    bases = [Path(f"/workspace/walinet/data/3T/VIDA_Vienna/{v}/{s}") for s in Res]

    process_subjects(bases, z=15, t=4)