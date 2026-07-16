# src/walinet/training_data/build_simulation_system.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import yaml

from walinet.config.build import (
    build_config,
)
from walinet.config.build_simulation import (
    build_simulation_config,
)
from walinet.config.schema import (
    TrainConfig,
)
from walinet.config.schema_simulation import (
    SimulationConfig,
)
from walinet.training_data.lcmodel_basis.acquisition import (
    PreparedBasis,
    prepare_basis_for_acquisition,
)
from walinet.training_data.metabolite_simulation import (
    MetaboliteSimulator,
)
from walinet.training_data.simulation_resources import (
    SimulationResources,
    build_simulation_resources,
)
from walinet.training_data.spectrum_simulator import (
    SpectrumSimulator,
)


@dataclass(frozen=True)
class SimulationSystem:
    """
    Fully constructed on-the-fly simulation system.

    The complete system is derived from one training configuration
    path:

        training config
            -> simulation config
                -> basis library
                -> metabolite profiles
            -> train/validation resource files
            -> ready-to-use simulators
    """

    train_config_path: Path
    simulation_config_path: Path

    train_config: TrainConfig
    simulation_config: SimulationConfig

    prepared_basis: PreparedBasis
    resources: SimulationResources

    metabolite_simulator: MetaboliteSimulator

    train_simulator: SpectrumSimulator
    validation_simulator: SpectrumSimulator

    device: torch.device


def _load_yaml_mapping(
    path: Path,
) -> dict:
    """
    Load one YAML file and require a top-level mapping.
    """
    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(
            "Configuration file not found:\n"
            f"  {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        raw = yaml.safe_load(
            file
        )

    if not isinstance(
        raw,
        dict,
    ):
        raise TypeError(
            "Configuration file must contain a YAML mapping:\n"
            f"  file:  {path}\n"
            f"  found: {type(raw)}"
        )

    return raw


def _resolve_device(
    train_cfg: TrainConfig,
) -> torch.device:
    """
    Derive the simulation device exclusively from run.gpu in the
    training configuration.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The training configuration requests a CUDA GPU, "
            "but CUDA is not available."
        )

    gpu_index = int(
        train_cfg.run.gpu
    )

    n_cuda_devices = torch.cuda.device_count()

    if not (
        0
        <= gpu_index
        < n_cuda_devices
    ):
        raise ValueError(
            "Configured GPU index is unavailable:\n"
            f"  requested: {gpu_index}\n"
            f"  available CUDA devices: {n_cuda_devices}"
        )

    return torch.device(
        f"cuda:{gpu_index}"
    )


def build_simulation_system(
    train_config_path: str | Path,
) -> SimulationSystem:
    """
    Construct the complete on-the-fly simulation system from one
    training configuration path.

    No simulation path, basis path, metabolite profile path,
    resource path, or device must be supplied separately.
    """
    train_config_path = Path(
        train_config_path
    ).expanduser().resolve()

    # ---------------------------------------------------------
    # Training configuration
    # ---------------------------------------------------------
    train_raw = _load_yaml_mapping(
        train_config_path
    )

    train_cfg = build_config(
        train_raw,
        config_dir=train_config_path.parent,
    )

    if train_cfg.data.source != "on_the_fly":
        raise ValueError(
            "build_simulation_system requires:\n"
            "  data.source: 'on_the_fly'"
        )

    if train_cfg.data.on_the_fly is None:
        raise ValueError(
            "The training configuration contains no "
            "data.on_the_fly section."
        )

    # build_config has already resolved this path relative to the
    # directory containing the training YAML.
    simulation_config_path = Path(
        train_cfg
        .data
        .on_the_fly
        .simulation_config
    ).resolve()

    # ---------------------------------------------------------
    # Simulation configuration
    # ---------------------------------------------------------
    simulation_raw = _load_yaml_mapping(
        simulation_config_path
    )

    simulation_cfg = build_simulation_config(
        simulation_raw,
        config_dir=simulation_config_path.parent,
    )

    # ---------------------------------------------------------
    # Device
    # ---------------------------------------------------------
    device = _resolve_device(
        train_cfg
    )

    torch.cuda.set_device(
        device
    )

    # ---------------------------------------------------------
    # LCModel basis
    # ---------------------------------------------------------
    prepared_basis = prepare_basis_for_acquisition(
        simulation_cfg.basis.library,
        target_bandwidth=(
            simulation_cfg
            .acquisition
            .bandwidth_hz
        ),
        target_n_timepoints=(
            simulation_cfg
            .acquisition
            .n_timepoints
        ),
        dataset_name="clean_fid",
    )

    # ---------------------------------------------------------
    # Water/lipid simulation resources
    # ---------------------------------------------------------
    resources_cpu = build_simulation_resources(
        train_cfg=train_cfg,
        simulation_cfg=simulation_cfg,
    )

    resources = resources_cpu.to(
        device
    )

    # ---------------------------------------------------------
    # Metabolite simulator
    # ---------------------------------------------------------
    metabolite_simulator = MetaboliteSimulator(
        prepared_basis=prepared_basis,
        config=simulation_cfg,
        device=device,
    )

    # ---------------------------------------------------------
    # Complete train and validation simulators
    # ---------------------------------------------------------
    train_simulator = SpectrumSimulator(
        pool=resources.train,
        metabolite_simulator=(
            metabolite_simulator
        ),
        config=simulation_cfg,
    )

    validation_simulator = SpectrumSimulator(
        pool=resources.validation,
        metabolite_simulator=(
            metabolite_simulator
        ),
        config=simulation_cfg,
    )

    return SimulationSystem(
        train_config_path=(
            train_config_path
        ),
        simulation_config_path=(
            simulation_config_path
        ),
        train_config=train_cfg,
        simulation_config=simulation_cfg,
        prepared_basis=prepared_basis,
        resources=resources,
        metabolite_simulator=(
            metabolite_simulator
        ),
        train_simulator=train_simulator,
        validation_simulator=(
            validation_simulator
        ),
        device=device,
    )