# src/walinet/config/build_training_data.py

from pathlib import Path

from .schema_training_data import (
    TrainingDataConfig,
    RunCfg,
    DataCfg,
    DataPathsCfg,
    WaterRemovalCfg,
    OutputCfg,
)


def validate_training_data_config(cfg: TrainingDataConfig) -> None:
    if cfg.run.seed < 0:
        raise ValueError("run.seed must be >= 0.")

    if cfg.data.base_dir == "":
        raise ValueError("data.base_dir must not be empty.")

    if len(cfg.data.subjects) == 0:
        raise ValueError("data.subjects must not be empty.")

    if cfg.data.paths.brain_mask == "":
        raise ValueError("data.paths.brain_mask must not be empty.")

    if cfg.data.paths.lipid_mask == "":
        raise ValueError("data.paths.lipid_mask must not be empty.")

    if cfg.data.paths.input_data == "":
        raise ValueError("data.paths.input_data must not be empty.")

    if cfg.data.paths.output_dir == "":
        raise ValueError("data.paths.output_dir must not be empty.")

    if cfg.water_removal.enabled:
        if cfg.water_removal.hsvd_components <= 0:
            raise ValueError("water_removal.hsvd_components must be > 0.")

        if cfg.water_removal.min_freq >= cfg.water_removal.max_freq:
            raise ValueError("water_removal.min_freq must be < water_removal.max_freq.")

        if cfg.water_removal.bandwidth <= 0:
            raise ValueError("water_removal.bandwidth must be > 0.")

        if cfg.water_removal.dwell_time <= 0:
            raise ValueError("water_removal.dwell_time must be > 0.")

        if cfg.water_removal.parallel_jobs <= 0:
            raise ValueError("water_removal.parallel_jobs must be > 0.")

        if cfg.water_removal.slice_batch_size <= 0:
            raise ValueError("water_removal.slice_batch_size must be > 0.")

    if cfg.output.version == "":
        raise ValueError("output.version must not be empty.")

    if cfg.output.isolated_water_filename == "":
        raise ValueError("output.isolated_water_filename must not be empty.")

    if "{version}" not in cfg.output.isolated_water_filename:
        raise ValueError("output.isolated_water_filename should contain '{version}'.")


def build_training_data_config(
    raw: dict,
    config_dir: Path | None = None,
) -> TrainingDataConfig:
    run_raw = raw.get("run", {})
    run = RunCfg(
        seed=int(run_raw.get("seed", 42)),
    )

    data_raw = raw["data"]
    paths_raw = data_raw["paths"]

    paths = DataPathsCfg(
        brain_mask=str(paths_raw["brain_mask"]),
        lipid_mask=str(paths_raw["lipid_mask"]),
        input_data=str(paths_raw["input_data"]),
        output_dir=str(paths_raw["output_dir"]),
    )

    base_dir = Path(data_raw["base_dir"])
    if config_dir is not None and not base_dir.is_absolute():
        base_dir = config_dir / base_dir

    data = DataCfg(
        base_dir=str(base_dir),
        subjects=list(data_raw["subjects"]),
        paths=paths,
    )

    water_raw = raw["water_removal"]

    bandwidth = float(water_raw["bandwidth"])
    dwell_time_raw = water_raw.get("dwell_time", None)
    dwell_time = 1.0 / bandwidth if dwell_time_raw is None else float(dwell_time_raw)

    water_removal = WaterRemovalCfg(
        enabled=bool(water_raw.get("enabled", True)),
        hsvd_components=int(water_raw["hsvd_components"]),
        min_freq=float(water_raw["min_freq"]),
        max_freq=float(water_raw["max_freq"]),
        bandwidth=bandwidth,
        dwell_time=dwell_time,
        parallel_jobs=int(water_raw.get("parallel_jobs", 1)),
        slice_batch_size=int(water_raw.get("slice_batch_size", 1)),
    )

    output_raw = raw["output"]

    output = OutputCfg(
        version=str(output_raw["version"]),
        isolated_water_filename=str(output_raw["isolated_water_filename"]),
        overwrite=bool(output_raw.get("overwrite", False)),
    )

    cfg = TrainingDataConfig(
        run=run,
        data=data,
        water_removal=water_removal,
        output=output,
    )

    validate_training_data_config(cfg)
    return cfg