# src/walinet/config/schema.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunCfg:
    name: str
    seed: int
    gpu: int


@dataclass(frozen=True)
class SimulationResourcesCfg:
    """
    Description of the subject-specific simulation-resource files.

    The filename is interpreted relative to each subject directory.
    The placeholder ``{version}`` is replaced by ``version``.
    """

    version: str
    filename: str


@dataclass(frozen=True)
class DataCfg:
    """
    Configuration for on-the-fly simulation.

    ``base_dir`` contains the subject directories.
    ``simulation_config`` points to the simulator YAML configuration.
    """

    base_dir: str
    train_subjects: list[str]
    val_subjects: list[str]

    simulation_config: str
    resources: SimulationResourcesCfg


@dataclass(frozen=True)
class OutputCfg:
    base_dir: str
    overwrite: bool


@dataclass(frozen=True)
class TrainingCfg:
    """
    Training configuration.

    ``n_batches`` determines how many newly simulated batches are
    used during each epoch.
    """

    batch_size: int
    epochs: int
    n_batches: int
    verbose: bool


@dataclass(frozen=True)
class ValidationCfg:
    """
    A finite synthetic validation dataset is generated once at startup
    using the specified seed and reused after every epoch.
    """

    seed: int
    n_spectra: int
    batch_size: int


@dataclass(frozen=True)
class OptimCfg:
    lr: float


@dataclass(frozen=True)
class SchedulerCfg:
    milestones: list[int]
    gamma: float


@dataclass(frozen=True)
class ModelCfg:
    architecture: str
    n_layers: int
    n_filters: int
    in_channels: int
    out_channels: int
    dropout: float


@dataclass(frozen=True)
class CheckpointCfg:
    """
    Optional warm start from the weights of an existing model.

    This loads model weights only. Optimizer state, scheduler state,
    and epoch number are not restored.
    """

    preload: bool
    preload_model: str


@dataclass(frozen=True)
class TrainConfig:
    run: RunCfg
    data: DataCfg
    output: OutputCfg
    training: TrainingCfg
    validation: ValidationCfg
    optim: OptimCfg
    scheduler: SchedulerCfg
    model: ModelCfg
    checkpoint: CheckpointCfg