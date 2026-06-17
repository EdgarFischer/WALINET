from pathlib import Path

from .schema import (
    TrainConfig,
    RunCfg,
    DataCfg,
    OutputCfg,
    TrainingCfg,
    OptimCfg,
    SchedulerCfg,
    ModelCfg,
    CheckpointCfg,
    PredictionCfg,
)


def _resolve_path(path: str, config_dir: Path | None) -> str:
    p = Path(path)

    if config_dir is not None and not p.is_absolute():
        p = config_dir / p

    return str(p)


def validate_config(cfg: TrainConfig) -> None:
    if cfg.run.name == "":
        raise ValueError("run.name must not be empty.")

    if cfg.run.seed < 0:
        raise ValueError("run.seed must be >= 0.")

    if cfg.run.gpu < 0:
        raise ValueError("run.gpu must be >= 0.")

    if cfg.data.base_dir == "":
        raise ValueError("data.base_dir must not be empty.")

    if len(cfg.data.train_subjects) == 0:
        raise ValueError("data.train_subjects must not be empty.")

    if len(cfg.data.val_subjects) == 0:
        raise ValueError("data.val_subjects must not be empty.")

    if cfg.data.version == "":
        raise ValueError("data.version must not be empty.")

    if cfg.data.train_data_filename == "":
        raise ValueError("data.train_data_filename must not be empty.")

    if "{version}" not in cfg.data.train_data_filename:
        raise ValueError("data.train_data_filename should contain '{version}'.")

    if cfg.output.base_dir == "":
        raise ValueError("output.base_dir must not be empty.")

    if cfg.training.batch_size <= 0:
        raise ValueError("training.batch_size must be > 0.")

    if cfg.training.num_workers < 0:
        raise ValueError("training.num_workers must be >= 0.")

    if cfg.training.epochs <= 0:
        raise ValueError("training.epochs must be > 0.")

    if cfg.training.n_batches == 0:
        raise ValueError("training.n_batches must be -1 or > 0.")

    if cfg.training.n_batches < -1:
        raise ValueError("training.n_batches must be -1 or > 0.")

    if cfg.training.n_val_batches == 0:
        raise ValueError("training.n_val_batches must be -1 or > 0.")

    if cfg.training.n_val_batches < -1:
        raise ValueError("training.n_val_batches must be -1 or > 0.")

    if cfg.optim.lr <= 0:
        raise ValueError("optim.lr must be > 0.")

    if len(cfg.scheduler.milestones) == 0:
        raise ValueError("scheduler.milestones must not be empty.")

    if cfg.scheduler.gamma <= 0:
        raise ValueError("scheduler.gamma must be > 0.")

    if cfg.model.n_layers <= 0:
        raise ValueError("model.n_layers must be > 0.")

    if cfg.model.n_filters <= 0:
        raise ValueError("model.n_filters must be > 0.")

    if cfg.model.in_channels <= 0:
        raise ValueError("model.in_channels must be > 0.")

    if cfg.model.out_channels <= 0:
        raise ValueError("model.out_channels must be > 0.")

    if cfg.model.dropout < 0:
        raise ValueError("model.dropout must be >= 0.")
    
    if cfg.data.normalization not in ["projection_energy", "max_abs"]:
        raise ValueError(
            "data.normalization must be 'projection_energy' or 'max_abs'."
        )


def build_config(
    raw: dict,
    config_dir: Path | None = None,
) -> TrainConfig:
    run_raw = raw["run"]

    run = RunCfg(
        name=str(run_raw["name"]),
        seed=int(run_raw.get("seed", 42)),
        gpu=int(run_raw["gpu"]),
    )

    data_raw = raw["data"]

    data = DataCfg(
        base_dir=_resolve_path(str(data_raw["base_dir"]), config_dir),
        train_subjects=list(data_raw["train_subjects"]),
        val_subjects=list(data_raw["val_subjects"]),
        version=str(data_raw["version"]),
        train_data_filename=str(
            data_raw.get("train_data_filename", "TrainData_{version}.h5")
        ),
        normalization=str(data_raw.get("normalization", "projection_energy")),
    )

    output_raw = raw["output"]

    output = OutputCfg(
        base_dir=_resolve_path(str(output_raw["base_dir"]), config_dir),
        overwrite=bool(output_raw.get("overwrite", False)),
    )

    training_raw = raw["training"]

    training = TrainingCfg(
        enabled=bool(training_raw.get("enabled", True)),
        batch_size=int(training_raw["batch_size"]),
        num_workers=int(training_raw.get("num_workers", 0)),
        epochs=int(training_raw["epochs"]),
        n_batches=int(training_raw.get("n_batches", -1)),
        n_val_batches=int(training_raw.get("n_val_batches", -1)),
        verbose=bool(training_raw.get("verbose", False)),
    )

    optim_raw = raw["optim"]

    optim = OptimCfg(
        lr=float(optim_raw["lr"]),
    )

    scheduler_raw = raw["scheduler"]

    scheduler = SchedulerCfg(
        milestones=[int(m) for m in scheduler_raw["milestones"]],
        gamma=float(scheduler_raw["gamma"]),
    )

    model_raw = raw["model"]

    model = ModelCfg(
        n_layers=int(model_raw["n_layers"]),
        n_filters=int(model_raw["n_filters"]),
        in_channels=int(model_raw["in_channels"]),
        out_channels=int(model_raw["out_channels"]),
        dropout=float(model_raw.get("dropout", 0.0)),
    )

    checkpoint_raw = raw.get("checkpoint", {})

    checkpoint = CheckpointCfg(
        preload=bool(checkpoint_raw.get("preload", False)),
        preload_model=str(checkpoint_raw.get("preload_model", "")),
    )

    prediction_raw = raw.get("prediction", {})

    prediction = PredictionCfg(
        enabled=bool(prediction_raw.get("enabled", False)),
    )

    cfg = TrainConfig(
        run=run,
        data=data,
        output=output,
        training=training,
        optim=optim,
        scheduler=scheduler,
        model=model,
        checkpoint=checkpoint,
        prediction=prediction,
    )

    validate_config(cfg)
    return cfg