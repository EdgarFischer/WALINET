import numpy as np
import h5py
import os

# Liste der Vol-IDs, die nacheinander verarbeitet werden sollen
target_volumes = [
    '4_Brisbane_B0_corrected',
    '5_B0_corrected',
    '6_B0_corrected',
    '7_B0_corrected',
    '8_B0_corrected',
    '9_B0_corrected'
    # Füge hier weitere Vol-IDs hinzu
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