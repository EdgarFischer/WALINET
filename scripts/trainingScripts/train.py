#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import argparse
import hashlib
import os
import random
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

YAML_SUFFIXES = {
    ".yaml",
    ".yml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default=str(
            ROOT
            / "configs"
            / "Training"
            / "train_7T.yaml"
        ),
        help="Path to training config YAML.",
    )

    return parser.parse_args()


def prepare_model_folder(
    *,
    model_dir: Path,
    overwrite: bool,
) -> None:
    """
    Create a clean model-output directory.
    """
    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(
                "Model directory already exists:\n"
                f"  {model_dir}\n\n"
                "Choose a different run.name or set "
                "output.overwrite: true."
            )

        shutil.rmtree(
            model_dir
        )

    (
        model_dir
        / "predictions"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        model_dir
        / "configs"
    ).mkdir(
        parents=True,
        exist_ok=True,
    )


def resolve_existing_file(
    path_value: str | Path,
    *,
    relative_to: Path,
) -> Path:
    """
    Resolve a potentially relative file path.

    Relative paths are first interpreted relative to the config
    containing the path. As a fallback, they are interpreted
    relative to the project root.
    """
    path = Path(
        path_value
    ).expanduser()

    if path.is_absolute():
        candidates = [
            path,
        ]
    else:
        candidates = [
            relative_to / path,
            ROOT / path,
        ]

    checked_paths: list[Path] = []

    for candidate in candidates:
        resolved = candidate.resolve()
        checked_paths.append(
            resolved
        )

        if resolved.is_file():
            return resolved

    checked_text = "\n".join(
        f"  {candidate}"
        for candidate in checked_paths
    )

    raise FileNotFoundError(
        "Referenced file does not exist. Checked:\n"
        f"{checked_text}"
    )


def resolve_output_base_dir(
    path_value: str | Path,
) -> Path:
    """
    Resolve the configured output base directory relative to the
    project root when it is not absolute.
    """
    path = Path(
        path_value
    ).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    return path.resolve()


def iter_string_values(
    value,
):
    """
    Recursively yield all strings contained in dictionaries,
    sequences, and nested configuration structures.
    """
    if isinstance(
        value,
        dict,
    ):
        for nested_value in value.values():
            yield from iter_string_values(
                nested_value
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        for nested_value in value:
            yield from iter_string_values(
                nested_value
            )

    elif isinstance(
        value,
        str,
    ):
        yield value


def resolve_yaml_reference(
    path_value: str,
    *,
    source_config_dir: Path,
) -> Path | None:
    """
    Resolve a YAML-looking string when it points to an existing
    file.

    Non-existing strings are returned as None so ordinary YAML
    values ending in similar text do not break the snapshot.
    """
    path = Path(
        path_value
    ).expanduser()

    if path.suffix.lower() not in YAML_SUFFIXES:
        return None

    if path.is_absolute():
        candidates = [
            path,
        ]
    else:
        candidates = [
            source_config_dir / path,
            ROOT / path,
        ]

    for candidate in candidates:
        resolved = candidate.resolve()

        if resolved.is_file():
            return resolved

    return None


def discover_yaml_dependencies(
    *,
    entry_config: Path,
    load_yaml_func,
) -> tuple[
    set[Path],
    list[tuple[Path, str]],
]:
    """
    Recursively discover YAML files referenced by another YAML
    file.

    This captures, for example:

    - the metabolite-distribution YAML,
    - nested profile YAMLs,
    - any further YAML configuration dependencies.
    """
    entry_config = (
        entry_config.resolve()
    )

    discovered: set[Path] = set()
    unresolved: list[
        tuple[Path, str]
    ] = []

    queue: list[Path] = [
        entry_config,
    ]

    while queue:
        current_config = (
            queue.pop(0).resolve()
        )

        if current_config in discovered:
            continue

        discovered.add(
            current_config
        )

        config_data = load_yaml_func(
            current_config
        )

        for string_value in iter_string_values(
            config_data
        ):
            candidate_path = Path(
                string_value
            )

            if (
                candidate_path
                .suffix
                .lower()
                not in YAML_SUFFIXES
            ):
                continue

            dependency = resolve_yaml_reference(
                string_value,
                source_config_dir=(
                    current_config.parent
                ),
            )

            if dependency is None:
                unresolved.append(
                    (
                        current_config,
                        string_value,
                    )
                )
                continue

            if dependency not in discovered:
                queue.append(
                    dependency
                )

    return (
        discovered,
        unresolved,
    )


def dependency_destination_path(
    *,
    source: Path,
    simulation_config_dir: Path,
) -> Path:
    """
    Generate a readable and collision-resistant relative path for
    a copied YAML dependency.
    """
    possible_roots = (
        simulation_config_dir,
        ROOT,
    )

    for root in possible_roots:
        try:
            return source.relative_to(
                root
            )
        except ValueError:
            pass

    digest = hashlib.sha1(
        str(source).encode(
            "utf-8"
        )
    ).hexdigest()[:10]

    return Path(
        f"{digest}_{source.name}"
    )


def snapshot_reproducibility_configs(
    *,
    model_dir: Path,
    training_config_path: Path,
    simulation_config_path: Path,
    load_yaml_func,
) -> None:
    """
    Copy the complete YAML configuration chain into the model
    directory.

    The copied files include:

    - exact training configuration,
    - exact simulation configuration,
    - recursively referenced YAML dependencies such as the
      metabolite-distribution configuration,
    - a manifest mapping original paths to copied paths.
    """
    configs_dir = (
        model_dir
        / "configs"
    )

    training_destination = (
        configs_dir
        / "training"
        / training_config_path.name
    )

    simulation_destination = (
        configs_dir
        / "simulation"
        / simulation_config_path.name
    )

    training_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    simulation_destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        training_config_path,
        training_destination,
    )

    shutil.copy2(
        simulation_config_path,
        simulation_destination,
    )

    (
        dependencies,
        unresolved_references,
    ) = discover_yaml_dependencies(
        entry_config=(
            simulation_config_path
        ),
        load_yaml_func=load_yaml_func,
    )

    copied_files: list[
        tuple[Path, Path]
    ] = [
        (
            training_config_path,
            training_destination,
        ),
        (
            simulation_config_path,
            simulation_destination,
        ),
    ]

    dependencies_dir = (
        configs_dir
        / "dependencies"
    )

    for dependency in sorted(
        dependencies
    ):
        if dependency == simulation_config_path:
            continue

        relative_destination = (
            dependency_destination_path(
                source=dependency,
                simulation_config_dir=(
                    simulation_config_path
                    .parent
                ),
            )
        )

        destination = (
            dependencies_dir
            / relative_destination
        )

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        shutil.copy2(
            dependency,
            destination,
        )

        copied_files.append(
            (
                dependency,
                destination,
            )
        )

    manifest_path = (
        configs_dir
        / "manifest.txt"
    )

    with open(
        manifest_path,
        "w",
    ) as manifest:
        manifest.write(
            "WALINET configuration snapshot\n"
        )
        manifest.write(
            "==============================\n\n"
        )

        manifest.write(
            "Copied YAML files\n"
        )
        manifest.write(
            "-----------------\n"
        )

        for source, destination in copied_files:
            manifest.write(
                f"Original: {source}\n"
            )
            manifest.write(
                "Snapshot: "
                f"{destination.relative_to(model_dir)}\n\n"
            )

        if unresolved_references:
            manifest.write(
                "\nUnresolved YAML-looking references\n"
            )
            manifest.write(
                "----------------------------------\n"
            )

            for source_config, reference in (
                unresolved_references
            ):
                manifest.write(
                    f"Config:    {source_config}\n"
                )
                manifest.write(
                    f"Reference: {reference}\n\n"
                )

    print(
        "Configuration snapshot created:"
    )

    for source, destination in copied_files:
        print(
            "  "
            f"{source.name} "
            "-> "
            f"{destination.relative_to(model_dir)}"
        )

    print(
        "  manifest "
        "-> "
        f"{manifest_path.relative_to(model_dir)}"
    )


def write_run_summary(
    *,
    model_dir: Path,
    config_path: Path,
    simulation_config_path: Path,
    cfg,
) -> None:
    """
    Store a compact, human-readable summary of the training run.
    """
    summary_path = (
        model_dir
        / "run_summary.txt"
    )

    with open(
        summary_path,
        "w",
    ) as file:
        file.write(
            f"run_name: {cfg.run.name}\n"
        )
        file.write(
            f"seed: {cfg.run.seed}\n"
        )
        file.write(
            f"configured_gpu: {cfg.run.gpu}\n"
        )
        file.write(
            f"architecture: {cfg.model.architecture}\n"
        )
        file.write(
            f"epochs: {cfg.training.epochs}\n"
        )
        file.write(
            f"batch_size: {cfg.training.batch_size}\n"
        )
        file.write(
            f"n_batches: {cfg.training.n_batches}\n"
        )
        file.write(
            f"learning_rate: {cfg.optim.lr}\n"
        )
        file.write(
            f"validation_mode: {cfg.validation.mode}\n"
        )
        file.write(
            f"validation_seed: {cfg.validation.seed}\n"
        )
        file.write(
            f"validation_n_spectra: "
            f"{cfg.validation.n_spectra}\n"
        )
        file.write(
            f"validation_batch_size: "
            f"{cfg.validation.batch_size}\n"
        )
        file.write(
            f"training_config: {config_path}\n"
        )
        file.write(
            "simulation_config: "
            f"{simulation_config_path}\n"
        )


if __name__ == "__main__":
    args = parse_args()

    sys.path.insert(
        0,
        str(SRC),
    )

    os.chdir(
        ROOT
    )

    from walinet.config.load import (
        load_yaml,
    )
    from walinet.config.build import (
        build_config,
    )

    config_path = resolve_existing_file(
        args.config,
        relative_to=ROOT,
    )

    config_dir = (
        config_path.parent
    )

    raw_config = load_yaml(
        config_path
    )

    cfg = build_config(
        raw_config,
        config_dir=config_dir,
    )

    if cfg.data.source != "on_the_fly":
        raise NotImplementedError(
            "The new trainer currently supports only "
            "data.source == 'on_the_fly'."
        )

    if cfg.data.on_the_fly is None:
        raise ValueError(
            "data.on_the_fly must be configured when "
            "data.source == 'on_the_fly'."
        )

    if cfg.validation.mode != "fixed_on_start":
        raise NotImplementedError(
            "The new trainer currently supports only "
            "validation.mode == 'fixed_on_start'."
        )

    if cfg.training.n_batches <= 0:
        raise ValueError(
            "training.n_batches must be > 0 for on-the-fly "
            "training."
        )

    if cfg.training.batch_size <= 0:
        raise ValueError(
            "training.batch_size must be > 0."
        )

    if cfg.validation.n_spectra <= 0:
        raise ValueError(
            "validation.n_spectra must be > 0."
        )

    if cfg.validation.batch_size <= 0:
        raise ValueError(
            "validation.batch_size must be > 0."
        )

    simulation_config_path = (
        resolve_existing_file(
            cfg.data
            .on_the_fly
            .simulation_config,
            relative_to=config_dir,
        )
    )

    # This must happen before importing torch.

    import torch
    import torch.nn as nn

    from torch.optim import Adam
    from torch.optim.lr_scheduler import (
        MultiStepLR,
    )

    from walinet.training.training import (
        create_fixed_validation_batches,
        train_one_epoch,
        validate_one_epoch,
    )

    from walinet.model.model import (
        yModel,
        uModel,
    )

    from walinet.training_data.build_simulation_system import (
        build_simulation_system,
    )

    # Global random seeds for model initialization and any
    # remaining Python/NumPy randomness.
    random.seed(
        cfg.run.seed
    )

    np.random.seed(
        cfg.run.seed
    )

    torch.manual_seed(
        cfg.run.seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            cfg.run.seed
        )

    architecture = str(
        cfg.model.architecture
    ).lower()

    if architecture not in {
        "ynet",
        "unet",
    }:
        raise ValueError(
            f"Unknown architecture {architecture!r}. "
            "Use 'ynet' or 'unet'."
        )

    output_base_dir = (
        resolve_output_base_dir(
            cfg.output.base_dir
        )
    )

    model_dir = (
        output_base_dir
        / cfg.run.name
    )

    prepare_model_folder(
        model_dir=model_dir,
        overwrite=cfg.output.overwrite,
    )

    snapshot_reproducibility_configs(
        model_dir=model_dir,
        training_config_path=config_path,
        simulation_config_path=(
            simulation_config_path
        ),
        load_yaml_func=load_yaml,
    )

    write_run_summary(
        model_dir=model_dir,
        config_path=config_path,
        simulation_config_path=(
            simulation_config_path
        ),
        cfg=cfg,
    )

    print(
        f"Model:     {cfg.run.name}"
    )
    print(
        f"Model dir: {model_dir}"
    )

    simulation_system = (
        build_simulation_system(
            config_path
        )
    )

    train_simulator = (
        simulation_system
        .train_simulator
    )

    device = torch.device(
        train_simulator.device
    )

    print(
        f"Using device: {device}"
    )
    print(
        "Simulator ready:"
    )
    print(
        f"  timepoints: "
        f"{train_simulator.n_timepoints}"
    )
    print(
        f"  device:     "
        f"{train_simulator.device}"
    )

    if architecture == "ynet":
        model = yModel(
            nLayers=(
                cfg.model.n_layers
            ),
            nFilters=(
                cfg.model.n_filters
            ),
            dropout=(
                cfg.model.dropout
            ),
            in_channels=(
                cfg.model.in_channels
            ),
            out_channels=(
                cfg.model.out_channels
            ),
        ).to(
            device
        )

    else:
        model = uModel(
            nLayers=(
                cfg.model.n_layers
            ),
            nFilters=(
                cfg.model.n_filters
            ),
            dropout=(
                cfg.model.dropout
            ),
            in_channels=(
                cfg.model.in_channels
            ),
            out_channels=(
                cfg.model.out_channels
            ),
        ).to(
            device
        )

    print(
        f"Using architecture: {architecture}"
    )

    if cfg.checkpoint.preload:
        preload_path = (
            output_base_dir
            / cfg.checkpoint.preload_model
            / "model_last.pt"
        )

        if not preload_path.is_file():
            raise FileNotFoundError(
                "Preload checkpoint does not exist:\n"
                f"  {preload_path}"
            )

        print(
            f"Loading model: {preload_path}"
        )

        model_state = torch.load(
            preload_path,
            map_location=device,
            weights_only=True,
        )

        model.load_state_dict(
            model_state
        )

    loss_func = nn.MSELoss()

    optimizer = Adam(
        model.parameters(),
        lr=cfg.optim.lr,
    )

    scheduler = MultiStepLR(
        optimizer,
        milestones=list(
            cfg.scheduler.milestones
        ),
        gamma=cfg.scheduler.gamma,
    )

    if cfg.training.enabled:
        train_generator = torch.Generator(
            device=device
        )

        train_generator.manual_seed(
            int(
                cfg.run.seed
            )
        )

        validation_generator = (
            torch.Generator(
                device=device
            )
        )

        validation_generator.manual_seed(
            int(
                cfg.validation.seed
            )
        )

        validation_batches = (
            create_fixed_validation_batches(
                simulator=train_simulator,
                generator=(
                    validation_generator
                ),
                n_spectra=(
                    cfg.validation
                    .n_spectra
                ),
                batch_size=(
                    cfg.validation
                    .batch_size
                ),
                architecture=architecture,
                verbose=(
                    cfg.training.verbose
                ),
            )
        )

        print(
            "Training configuration:"
        )
        print(
            f"  epochs:                "
            f"{cfg.training.epochs}"
        )
        print(
            f"  batches per epoch:     "
            f"{cfg.training.n_batches}"
        )
        print(
            f"  training batch size:   "
            f"{cfg.training.batch_size}"
        )
        print(
            f"  spectra per epoch:     "
            f"{cfg.training.n_batches * cfg.training.batch_size}"
        )
        print(
            f"  validation spectra:    "
            f"{cfg.validation.n_spectra}"
        )
        print(
            f"  validation batch size: "
            f"{cfg.validation.batch_size}"
        )
        print(
            f"  validation batches:    "
            f"{len(validation_batches)}"
        )

        best_loss: float | None = None

        loss_path = (
            model_dir
            / "loss.txt"
        )

        with open(
            loss_path,
            "w",
        ) as file:
            file.write(
                "Epoch; Epoch Loss; "
                "Validation Loss; "
                "Learning Rate;\n"
            )

        for epoch in range(
            cfg.training.epochs
        ):
            model, train_loss = (
                train_one_epoch(
                    model=model,
                    simulator=(
                        train_simulator
                    ),
                    generator=(
                        train_generator
                    ),
                    optimizer=optimizer,
                    loss_func=loss_func,
                    architecture=(
                        architecture
                    ),
                    batch_size=(
                        cfg.training
                        .batch_size
                    ),
                    n_batches=(
                        cfg.training
                        .n_batches
                    ),
                    verbose=(
                        cfg.training
                        .verbose
                    ),
                    device=device,
                    epoch=epoch,
                )
            )

            val_loss = (
                validate_one_epoch(
                    model=model,
                    validation_batches=(
                        validation_batches
                    ),
                    loss_func=loss_func,
                    architecture=(
                        architecture
                    ),
                    verbose=(
                        cfg.training
                        .verbose
                    ),
                    device=device,
                    epoch=epoch,
                )
            )

            learning_rate = (
                scheduler
                .get_last_lr()[0]
            )

            torch.save(
                model.state_dict(),
                model_dir
                / "model_last.pt",
            )

            is_best_model = (
                best_loss is None
                or val_loss < best_loss
            )

            if is_best_model:
                best_loss = val_loss

                torch.save(
                    model.state_dict(),
                    model_dir
                    / "model_best.pt",
                )

            with open(
                loss_path,
                "a",
            ) as file:
                log = (
                    "Epoch: {:03d}, "
                    "Loss: {:.10f}, "
                    "Val Loss: {:.10f}, "
                    "LR: {:.10f}"
                )

                file.write(
                    log.format(
                        epoch + 1,
                        train_loss,
                        val_loss,
                        learning_rate,
                    )
                )

                if is_best_model:
                    file.write(
                        ", best model"
                    )

                file.write(
                    "\n"
                )

            scheduler.step()

        print(
            "Simulation retry statistics:"
        )
        print(
            f"  discarded batches: "
            f"{train_simulator.discarded_batches}"
        )
        print(
            f"  discarded spectra: "
            f"{train_simulator.discarded_spectra}"
        )

    if cfg.prediction.enabled:
        raise NotImplementedError(
            "Prediction should be run through a separate "
            "prediction script."
        )

    print(
        "All done!"
    )