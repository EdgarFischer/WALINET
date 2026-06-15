# src/walinet/config/schema_training_data.py

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RunCfg:
    seed: int


@dataclass(frozen=True)
class DataPathsCfg:
    brain_mask: str
    lipid_mask: str
    input_data: str
    output_dir: str


@dataclass(frozen=True)
class DataCfg:
    base_dir: str
    subjects: List[str]
    paths: DataPathsCfg


@dataclass(frozen=True)
class WaterRemovalCfg:
    enabled: bool
    hsvd_components: int
    min_freq: float
    max_freq: float
    bandwidth: float
    dwell_time: float
    parallel_jobs: int
    slice_batch_size: int


@dataclass(frozen=True)
class OutputCfg:
    version: str
    isolated_water_filename: str
    overwrite: bool


@dataclass(frozen=True)
class TrainingDataConfig:
    run: RunCfg
    data: DataCfg
    water_removal: WaterRemovalCfg
    output: OutputCfg