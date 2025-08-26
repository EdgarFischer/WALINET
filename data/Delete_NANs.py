import numpy as np
import h5py
import os

# Liste der Vol-IDs, die nacheinander verarbeitet werden sollen
target_volumes = [
    '1_London_B0corrected_wo_LipidMask',
    '2_London_B0corrected_wo_LipidMask',
    '3_London_B0corrected_wo_LipidMask',
    '4_London_B0corrected_wo_LipidMask',
    '5_London_B0corrected_wo_LipidMask',
    '1_Brisbane_B0corrected_wo_LipidMask',
    '3_Brisbane_B0corrected_wo_LipidMask',
    '4_Brisbane_B0corrected_wo_LipidMask',
    '5_Brisbane_B0corrected_wo_LipidMask',
    '6_Brisbane_B0corrected_wo_LipidMask',
    '5_B0corrected_wo_LipidMask',
    '6_B0corrected_wo_LipidMask',
    '7_B0corrected_wo_LipidMask',
    '8_B0corrected_wo_LipidMask',
    '9_B0corrected_wo_LipidMask'
]

# Basis-Pfade und Dateinamen-Versionen
version_src = '1.0'
version_dst = '1.0_noNans'
datasets_to_clean = {'lipid', 'lipid_proj', 'metab', 'spectra', 'water'}

for Vol in target_volumes:
    base = f'Vol{Vol}/TrainData'
    src = os.path.join(base, f'TrainData_v_{version_src}.h5')
    dst = os.path.join(base, f'TrainData_v_{version_dst}.h5')

    if not os.path.isfile(src):
        print(f"❌ Quelldatei nicht gefunden: {src}")
        continue

    # 1) Einlesen und fehlerhafte Zeilen identifizieren
    with h5py.File(src, 'r') as f:
        spectra = f['spectra'][...]
    bad_rows = np.unique(np.argwhere(~np.isfinite(spectra))[:, 0])
    keep_mask = ~np.isin(np.arange(spectra.shape[0]), bad_rows)
    print(f"Verarbeite {Vol}:")
    print(f"  Gefundene fehlerhafte Zeilen: {bad_rows.tolist()}")
    print(f"  Behalte {keep_mask.sum()} von {spectra.shape[0]} Zeilen")

    # 2) Kopieren und Bereinigen
    with h5py.File(src, 'r') as fin, h5py.File(dst, 'w') as fout:
        for key in fin.keys():
            data = fin[key][...]
            if key in datasets_to_clean:
                data = data[keep_mask]
            fout.create_dataset(key, data=data, compression='gzip')

    print(f"✅ Cleaned file written to {dst}\n")