# src/walinet/config/schema.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RunCfg:
    name: str
    seed: int
    gpu: int


@dataclass(frozen=True)
class OnTheFlyResourcesCfg:
    """
    Description of the subject-specific simulation-resource files.

    The filename is interpreted relative to each subject directory.
    """

    version: str
    filename: str


@dataclass(frozen=True)
class OnTheFlyCfg:
    """
    Configuration used only when data.source == "on_the_fly".
    """

    simulation_config: str
    resources: OnTheFlyResourcesCfg


@dataclass(frozen=True)
class PrecomputedCfg:
    """
    Configuration used only when data.source == "precomputed".
    """

    version: str
    train_data_filename: str


@dataclass(frozen=True)
class DataCfg:
    source: str

    base_dir: str
    train_subjects: list[str]
    val_subjects: list[str]

    normalization: str

    on_the_fly: OnTheFlyCfg | None
    precomputed: PrecomputedCfg | None


@dataclass(frozen=True)
class OutputCfg:
    base_dir: str
    overwrite: bool


@dataclass(frozen=True)
class TrainingCfg:
    enabled: bool
    batch_size: int
    num_workers: int
    epochs: int
    n_batches: int
    verbose: bool


@dataclass(frozen=True)
class ValidationCfg:
    """
    Validation dataset configuration.

    For mode == "fixed_on_start":
        A finite synthetic validation dataset is generated once
        with the specified seed and reused after every epoch.

    For mode == "precomputed":
        Validation samples are read from precomputed files.
        n_spectra == -1 means that all available samples are used.
    """

    mode: str
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
    preload: bool
    preload_model: str


@dataclass(frozen=True)
class PredictionCfg:
    enabled: bool


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
    prediction: PredictionCfg