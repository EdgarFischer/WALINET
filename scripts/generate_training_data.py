#!/usr/bin/env python3

from pathlib import Path
import sys
import argparse
import shutil
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "generate_training_data_3T.yaml"),
        help="Path to training-data generation config YAML.",
    )
    return p.parse_args()


def copy_config_to_subject_train_data(config_path: Path, cfg, subject: str) -> None:
    subject_train_dir = (
        Path(cfg.data.base_dir)
        / subject
        / cfg.data.paths.output_dir
    )

    config_dir = subject_train_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"generate_training_data_{cfg.output.version}.yaml"
    shutil.copy(config_path, config_dir / out_name)


if __name__ == "__main__":
    args = parse_args()

    sys.path.insert(0, str(SRC))

    from walinet.config.load import load_yaml
    from walinet.config.build_training_data import build_training_data_config
    from walinet.training_data.water_removal import get_or_create_isolated_water
    from walinet.training_data.simulation import process_subject

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    raw = load_yaml(config_path)
    cfg = build_training_data_config(raw, config_dir=config_dir)

    np.random.seed(cfg.run.seed)
    rng = np.random

    sampling_rate = cfg.acquisition.bandwidth

    for subject in cfg.data.subjects:
        print(f"\n=== Subject: {subject} ===")

        copy_config_to_subject_train_data(config_path, cfg, subject)

        # Loads existing isolated water or computes it if missing.
        get_or_create_isolated_water(subject, cfg)

        # Aborts internally if TrainData_{version}.h5 already exists.
        process_subject(
            sub=subject,
            path=cfg.data.base_dir,
            version=cfg.output.version,
            rng=rng,
            n_spectra=cfg.simulation.n_spectra,
            n_random_lipid=cfg.simulation.lipid.n_random_lipid,
            max_lipid_scaling=cfg.simulation.lipid.max_scaling,
            min_snr=cfg.simulation.metabolite.min_snr,
            max_snr=cfg.simulation.metabolite.max_snr,
            n_timepoints=cfg.acquisition.n_timepoints,
            sampling_rate=sampling_rate,
            nmr_freq=cfg.acquisition.nmr_freq,
            max_freq_shift=cfg.simulation.metabolite.max_freq_shift,
            min_peak_width=cfg.simulation.metabolite.min_peak_width,
            max_peak_width=cfg.simulation.metabolite.max_peak_width,
            max_acqu_delay=cfg.simulation.metabolite.max_acqu_delay,
            water_scaling_min=cfg.simulation.water.scaling_min,
            water_scaling_max=cfg.simulation.water.scaling_max,
            mean_std_path=cfg.simulation.metabolite.mean_std_path,
            modes_glob=cfg.simulation.metabolite.modes_glob,
            lipid_projection_target=cfg.lipid_projection.target,
            lipid_projection_tol=cfg.lipid_projection.tol,
            lipid_projection_max_iter=cfg.lipid_projection.max_iter,
        )

    print("\nAll done!")