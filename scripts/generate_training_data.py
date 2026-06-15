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

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    raw = load_yaml(config_path)
    cfg = build_training_data_config(raw, config_dir=config_dir)

    np.random.seed(cfg.run.seed)

    for subject in cfg.data.subjects:
        copy_config_to_subject_train_data(config_path, cfg, subject)

        water_rrrt = get_or_create_isolated_water(subject, cfg)

        # Later:
        # lipid_result = get_or_create_lipid_result(subject, cfg, water_rrrt)
        # training_data = generate_training_samples(subject, cfg, water_rrrt, lipid_result)

    print("ENDE!")