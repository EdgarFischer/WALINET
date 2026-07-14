# src/walinet/config/schema_water_lipid.py

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WaterLipidDataPathsCfg:
    brain_mask: str
    lipid_mask: str
    input_data: str
    output_dir: str


@dataclass(frozen=True)
class WaterLipidDataCfg:
    base_dir: str
    subjects: list[str]
    paths: WaterLipidDataPathsCfg


@dataclass(frozen=True)
class WaterExtractionCfg:
    bandwidth: float
    hsvd_components: int
    min_freq: float
    max_freq: float
    parallel_jobs: int
    slice_batch_size: int

    @property
    def dwell_time(self) -> float:
        """
        Time between consecutive FID samples in seconds.
        """
        return 1.0 / self.bandwidth


@dataclass(frozen=True)
class WaterLipidResourcesCfg:
    simulation_resources_filename: str
    overwrite: bool


@dataclass(frozen=True)
class LipidProjectionCfg:
    enabled: bool
    n_timepoints: list[int]
    target: float
    tol: float
    max_iter: int


@dataclass(frozen=True)
class WaterLipidExtractionConfig:
    version: str
    data: WaterLipidDataCfg
    water_extraction: WaterExtractionCfg
    resources: WaterLipidResourcesCfg
    lipid_projection: LipidProjectionCfg