from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RunCfg:
    name: str
    seed: int
    gpu: int


@dataclass(frozen=True)
class DataCfg:
    base_dir: str
    train_subjects: List[str]
    val_subjects: List[str]
    version: str
    train_data_filename: str


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
    n_val_batches: int
    verbose: bool


@dataclass(frozen=True)
class OptimCfg:
    lr: float


@dataclass(frozen=True)
class SchedulerCfg:
    milestones: List[int]
    gamma: float


@dataclass(frozen=True)
class ModelCfg:
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
    optim: OptimCfg
    scheduler: SchedulerCfg
    model: ModelCfg
    checkpoint: CheckpointCfg
    prediction: PredictionCfg