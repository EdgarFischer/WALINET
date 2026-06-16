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
class AcquisitionCfg:
    bandwidth: float
    n_timepoints: int
    nmr_freq: float


@dataclass(frozen=True)
class LipidProjectionCfg:
    target: float
    tol: float
    max_iter: int


@dataclass(frozen=True)
class SimulationMetaboliteCfg:
    mean_std_path: str
    modes_glob: str
    min_snr: float
    max_snr: float
    max_freq_shift: float
    min_peak_width: float
    max_peak_width: float
    max_acqu_delay: float


@dataclass(frozen=True)
class SimulationLipidCfg:
    n_random_lipid: int
    max_scaling: float


@dataclass(frozen=True)
class SimulationWaterCfg:
    scaling_min: float
    scaling_max: float


@dataclass(frozen=True)
class SimulationCfg:
    n_spectra: int
    metabolite: SimulationMetaboliteCfg
    lipid: SimulationLipidCfg
    water: SimulationWaterCfg


@dataclass(frozen=True)
class OutputCfg:
    version: str
    isolated_water_filename: str
    train_data_filename: str
    overwrite: bool


@dataclass(frozen=True)
class TrainingDataConfig:
    run: RunCfg
    data: DataCfg
    water_removal: WaterRemovalCfg
    acquisition: AcquisitionCfg
    lipid_projection: LipidProjectionCfg
    simulation: SimulationCfg
    output: OutputCfg