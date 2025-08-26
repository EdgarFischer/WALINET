#!/usr/bin/env bash
set -euo pipefail

# Quell-Ordner OHNE Suffix
volumes=(
  #Vol1_London
  #Vol2_London
  #Vol3_London
  #Vol4_London
  Vol6_Brisbane
)

for dir in "${volumes[@]}"; do
  if [[ ! -d "$dir" ]]; then
    echo "⚠︎ Verzeichnis nicht gefunden: $dir – übersprungen"
    continue
  fi

  # Ziel: _B0_corrected statt _nextstep
  new="${dir}_B0_corrected"

  # Zielstruktur anlegen
  mkdir -p "$new"/{masks,OriginalData,TrainData}

  # masks komplett kopieren
  if [[ -d "$dir/masks" ]]; then
    cp -a "$dir/masks/." "$new/masks/"
  else
    echo "⚠︎ $dir/masks nicht gefunden – übersprungen"
  fi

  # korrigierte Datei umbenannt kopieren (data_B0corrected.npy -> data.npy)
  src="$dir/OriginalData/data_B0corrected.npy"
  dst="$new/OriginalData/data.npy"
  if [[ -f "$src" ]]; then
    # -n: nicht überschreiben, falls bereits vorhanden
    cp -n "$src" "$dst" || echo "ℹ︎ $dst existiert bereits – nicht überschrieben"
  else
    echo "⚠︎ $src nicht gefunden – übersprungen"
  fi

  echo "✅ Verarbeitung von $dir abgeschlossen → $new"
done


