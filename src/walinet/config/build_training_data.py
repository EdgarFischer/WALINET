# src/walinet/config/build_training_data.py

from pathlib import Path

from .schema_training_data import (
    TrainingDataConfig,
    RunCfg,
    DataCfg,
    DataPathsCfg,
    WaterRemovalCfg,
    AcquisitionCfg,
    LipidProjectionCfg,
    SimulationCfg,
    SimulationMetaboliteCfg,
    SimulationLipidCfg,
    SimulationWaterCfg,
    OutputCfg,
)


def _resolve_path(path: str, config_dir: Path | None) -> str:
    p = Path(path)

    if config_dir is not None and not p.is_absolute():
        p = config_dir / p

    return str(p)


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

    if cfg.acquisition.bandwidth <= 0:
        raise ValueError("acquisition.bandwidth must be > 0.")

    if cfg.acquisition.n_timepoints <= 0:
        raise ValueError("acquisition.n_timepoints must be > 0.")

    if cfg.acquisition.nmr_freq <= 0:
        raise ValueError("acquisition.nmr_freq must be > 0.")

    if not (0 < cfg.lipid_projection.target <= 1):
        raise ValueError("lipid_projection.target must be in (0, 1].")

    if cfg.lipid_projection.tol <= 0:
        raise ValueError("lipid_projection.tol must be > 0.")

    if cfg.lipid_projection.max_iter <= 0:
        raise ValueError("lipid_projection.max_iter must be > 0.")

    if cfg.simulation.n_spectra <= 0:
        raise ValueError("simulation.n_spectra must be > 0.")

    if cfg.simulation.metabolite.mean_std_path == "":
        raise ValueError("simulation.metabolite.mean_std_path must not be empty.")

    if cfg.simulation.metabolite.modes_glob == "":
        raise ValueError("simulation.metabolite.modes_glob must not be empty.")

    if cfg.simulation.metabolite.min_snr <= 0:
        raise ValueError("simulation.metabolite.min_snr must be > 0.")

    if cfg.simulation.metabolite.max_snr < cfg.simulation.metabolite.min_snr:
        raise ValueError(
            "simulation.metabolite.max_snr must be >= simulation.metabolite.min_snr."
        )

    if cfg.simulation.metabolite.max_freq_shift < 0:
        raise ValueError("simulation.metabolite.max_freq_shift must be >= 0.")

    if cfg.simulation.metabolite.min_peak_width < 0:
        raise ValueError("simulation.metabolite.min_peak_width must be >= 0.")

    if cfg.simulation.metabolite.max_peak_width < cfg.simulation.metabolite.min_peak_width:
        raise ValueError(
            "simulation.metabolite.max_peak_width must be >= "
            "simulation.metabolite.min_peak_width."
        )

    if cfg.simulation.metabolite.max_acqu_delay < 0:
        raise ValueError("simulation.metabolite.max_acqu_delay must be >= 0.")

    if cfg.simulation.lipid.n_random_lipid <= 0:
        raise ValueError("simulation.lipid.n_random_lipid must be > 0.")

    if cfg.simulation.lipid.max_scaling <= 0:
        raise ValueError("simulation.lipid.max_scaling must be > 0.")

    if cfg.simulation.water.scaling_min < 0:
        raise ValueError("simulation.water.scaling_min must be >= 0.")

    if cfg.simulation.water.scaling_max < cfg.simulation.water.scaling_min:
        raise ValueError(
            "simulation.water.scaling_max must be >= simulation.water.scaling_min."
        )

    if cfg.output.version == "":
        raise ValueError("output.version must not be empty.")

    if cfg.output.isolated_water_filename == "":
        raise ValueError("output.isolated_water_filename must not be empty.")

    if "{version}" not in cfg.output.isolated_water_filename:
        raise ValueError("output.isolated_water_filename should contain '{version}'.")

    if cfg.output.train_data_filename == "":
        raise ValueError("output.train_data_filename must not be empty.")

    if "{version}" not in cfg.output.train_data_filename:
        raise ValueError("output.train_data_filename should contain '{version}'.")


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

    water_bandwidth = float(water_raw["bandwidth"])
    dwell_time_raw = water_raw.get("dwell_time", None)
    dwell_time = (
        1.0 / water_bandwidth
        if dwell_time_raw is None
        else float(dwell_time_raw)
    )

    water_removal = WaterRemovalCfg(
        enabled=bool(water_raw.get("enabled", True)),
        hsvd_components=int(water_raw["hsvd_components"]),
        min_freq=float(water_raw["min_freq"]),
        max_freq=float(water_raw["max_freq"]),
        bandwidth=water_bandwidth,
        dwell_time=dwell_time,
        parallel_jobs=int(water_raw.get("parallel_jobs", 1)),
        slice_batch_size=int(water_raw.get("slice_batch_size", 1)),
    )

    acquisition_raw = raw["acquisition"]

    acquisition = AcquisitionCfg(
        bandwidth=float(acquisition_raw["bandwidth"]),
        n_timepoints=int(acquisition_raw["n_timepoints"]),
        nmr_freq=float(acquisition_raw["nmr_freq"]),
    )

    lipid_projection_raw = raw["lipid_projection"]

    lipid_projection = LipidProjectionCfg(
        target=float(lipid_projection_raw["target"]),
        tol=float(lipid_projection_raw["tol"]),
        max_iter=int(lipid_projection_raw["max_iter"]),
    )

    simulation_raw = raw["simulation"]

    metabolite_raw = simulation_raw["metabolite"]
    metabolite = SimulationMetaboliteCfg(
        mean_std_path=_resolve_path(
            str(metabolite_raw["mean_std_path"]),
            config_dir,
        ),
        modes_glob=_resolve_path(
            str(metabolite_raw["modes_glob"]),
            config_dir,
        ),
        min_snr=float(metabolite_raw["min_snr"]),
        max_snr=float(metabolite_raw["max_snr"]),
        max_freq_shift=float(metabolite_raw["max_freq_shift"]),
        min_peak_width=float(metabolite_raw["min_peak_width"]),
        max_peak_width=float(metabolite_raw["max_peak_width"]),
        max_acqu_delay=float(metabolite_raw["max_acqu_delay"]),
    )

    lipid_raw = simulation_raw["lipid"]
    lipid = SimulationLipidCfg(
        n_random_lipid=int(lipid_raw["n_random_lipid"]),
        max_scaling=float(lipid_raw["max_scaling"]),
    )

    water_sim_raw = simulation_raw["water"]
    water = SimulationWaterCfg(
        scaling_min=float(water_sim_raw["scaling_min"]),
        scaling_max=float(water_sim_raw["scaling_max"]),
    )

    simulation = SimulationCfg(
        n_spectra=int(simulation_raw["n_spectra"]),
        metabolite=metabolite,
        lipid=lipid,
        water=water,
    )

    output_raw = raw["output"]

    output = OutputCfg(
        version=str(output_raw["version"]),
        isolated_water_filename=str(output_raw["isolated_water_filename"]),
        train_data_filename=str(
            output_raw.get("train_data_filename", "TrainData_{version}.h5")
        ),
        overwrite=bool(output_raw.get("overwrite", False)),
    )

    cfg = TrainingDataConfig(
        run=run,
        data=data,
        water_removal=water_removal,
        acquisition=acquisition,
        lipid_projection=lipid_projection,
        simulation=simulation,
        output=output,
    )

    validate_training_data_config(cfg)
    return cfg