import sys
from pathlib import Path

WALINET_ROOT = Path(
    "/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/public/hfish/walinet"
)

sys.path.insert(0, str(WALINET_ROOT / "src"))

from walinet.data.dataprep import process_subjects

bases = [
    WALINET_ROOT / "data/7T/NoB0Correction/Lukas",
]

results = process_subjects(
    bases=bases,
    z=15,
    t=4,
)

for subject_path, result in results.items():
    print(subject_path, "->", result)