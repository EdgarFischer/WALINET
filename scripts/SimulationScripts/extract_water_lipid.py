#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys


# Allow execution without an editable package installation.
project_root = (
    Path(__file__)
    .resolve()
    .parents[2]
)

src_path = (
    project_root
    / "src"
)

if str(src_path) not in sys.path:
    sys.path.insert(
        0,
        str(src_path),
    )


from walinet.config.build_water_lipid import (
    build_water_lipid_extraction_config,
)
from walinet.config.load import (
    load_yaml,
)
from walinet.training_data.water_lipid_extraction import (
    process_all_subjects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract isolated water and "
            "frequency-domain water/lipid "
            "simulation resources."
        )
    )

    parser.add_argument(
        "config",
        type=Path,
        help=(
            "Path to the water/lipid "
            "extraction YAML configuration."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config_path = (
        args.config
        .expanduser()
        .resolve()
    )

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: "
            f"{config_path}"
        )

    raw_config = load_yaml(
        config_path
    )

    cfg = (
        build_water_lipid_extraction_config(
            raw_config,
            config_dir=(
                config_path.parent
            ),
        )
    )

    print(
        "Water/lipid extraction configuration"
    )
    print(
        f"  Config   : {config_path}"
    )
    print(
        f"  Version  : {cfg.version}"
    )
    print(
        f"  Base dir : {cfg.data.base_dir}"
    )
    print(
        f"  Subjects : "
        f"{len(cfg.data.subjects)}"
    )
    print(
        f"  Bandwidth: "
        f"{cfg.water_extraction.bandwidth} Hz"
    )
    print(
        f"  FID length: native/full"
    )

    process_all_subjects(
        cfg
    )


if __name__ == "__main__":
    main()