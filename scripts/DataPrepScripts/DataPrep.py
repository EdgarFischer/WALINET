import sys
import os
from pathlib import Path
sys.path.append(os.path.abspath("../../src"))
sys.path.append(os.path.abspath(".."))
from walinet.data.dataprep import * 

subjects = [
    "MS_180"
]

bases = [Path(f"/workspace/walinet/data/7T/NoB0Correction/{s}") for s in subjects]

process_subjects(bases, z=15, t=4)