#!/usr/bin/env bash
set -euo pipefail

# Liste der Quell-Ordner (ohne _B0_corrected)
volumes=(
  Vol1_London
  Vol2_London
  Vol3_London
  Vol4_London
  Vol5_London
)

for dir in "${volumes[@]}"; do
  # Quell-Verzeichnis prüfen
  if [[ ! -d "$dir" ]]; then
    echo "⚠︎ Verzeichnis nicht gefunden: $dir – übersprungen"
    continue
  fi

  # Ziel-Ordner mit _B0_corrected
  new="${dir}_B0_corrected"

  # Unterordnerstruktur anlegen
  mkdir -p "$new"/{masks,OriginalData,TrainData}

  # 1) masks komplett kopieren
  if [[ -d "$dir/masks" ]]; then
    cp -a "$dir/masks/." "$new/masks/"
  else
    echo "⚠︎ $dir/masks nicht gefunden – übersprungen"
  fi

  # 2) korrigierte Datei aus OriginalData kopieren
  src="$dir/OriginalData/data_B0corrected.npy"
  dst="$new/OriginalData/data.npy"
  if [[ -f "$src" ]]; then
    cp "$src" "$dst"
  else
    echo "⚠︎ $src nicht gefunden – übersprungen"
  fi

  # 3) TrainData bleibt leer (mkdir), kann hier bei Bedarf befüllt werden

  echo "✅ Verarbeitung abgeschlossen: $dir → $new"
done


