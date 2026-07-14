# src/walinet/config/build.py

from __future__ import annotations

from pathlib import Path

from .schema import (
    CheckpointCfg,
    DataCfg,
    ModelCfg,
    OnTheFlyCfg,
    OnTheFlyResourcesCfg,
    OptimCfg,
    OutputCfg,
    PredictionCfg,
    PrecomputedCfg,
    RunCfg,
    SchedulerCfg,
    TrainConfig,
    TrainingCfg,
    ValidationCfg,
)


VALID_DATA_SOURCES = {
    "on_the_fly",
    "precomputed",
}

VALID_NORMALIZATIONS = {
    "projection_energy",
    "max_abs",
}

VALID_VALIDATION_MODES = {
    "fixed_on_start",
    "precomputed",
}

VALID_MODEL_ARCHITECTURES = {
    "unet",
    "ynet",
}


def _resolve_path(
    path: str,
    config_dir: Path | None,
) -> str:
    """
    Resolve a filesystem path relative to the directory containing
    the training YAML.

    Empty paths remain empty.
    """
    path = path.strip()

    if not path:
        return ""

    resolved_path = Path(
        path
    ).expanduser()

    if (
        config_dir is not None
        and not resolved_path.is_absolute()
    ):
        resolved_path = (
            config_dir
            / resolved_path
        )

    return str(
        resolved_path.resolve()
    )


def _validate_subjects(
    *,
    name: str,
    subjects: list[str],
) -> None:
    """
    Validate one subject list.
    """
    if not subjects:
        raise ValueError(
            f"{name} must not be empty."
        )

    empty_indices = [
        index
        for index, subject in enumerate(subjects)
        if not subject.strip()
    ]

    if empty_indices:
        raise ValueError(
            f"{name} contains empty entries at "
            f"indices {empty_indices}."
        )

    if len(subjects) != len(set(subjects)):
        raise ValueError(
            f"{name} contains duplicate entries."
        )


def validate_config(
    cfg: TrainConfig,
) -> None:
    """
    Validate the complete training configuration.
    """

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------
    if not cfg.run.name.strip():
        raise ValueError(
            "run.name must not be empty."
        )

    if cfg.run.seed < 0:
        raise ValueError(
            "run.seed must be >= 0."
        )

    if cfg.run.gpu < 0:
        raise ValueError(
            "run.gpu must be >= 0."
        )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    if cfg.data.source not in VALID_DATA_SOURCES:
        raise ValueError(
            "data.source must be one of "
            f"{sorted(VALID_DATA_SOURCES)}, "
            f"but found {cfg.data.source!r}."
        )

    if not cfg.data.base_dir.strip():
        raise ValueError(
            "data.base_dir must not be empty."
        )

    _validate_subjects(
        name="data.train_subjects",
        subjects=cfg.data.train_subjects,
    )

    _validate_subjects(
        name="data.val_subjects",
        subjects=cfg.data.val_subjects,
    )

    overlapping_subjects = sorted(
        set(cfg.data.train_subjects)
        & set(cfg.data.val_subjects)
    )

    if overlapping_subjects:
        raise ValueError(
            "Training and validation subjects must be "
            "disjoint. Overlapping subjects:\n"
            f"  {overlapping_subjects}"
        )

    if (
        cfg.data.normalization
        not in VALID_NORMALIZATIONS
    ):
        raise ValueError(
            "data.normalization must be one of "
            f"{sorted(VALID_NORMALIZATIONS)}, "
            f"but found {cfg.data.normalization!r}."
        )

    # ---------------------------------------------------------
    # On-the-fly data
    # ---------------------------------------------------------
    if cfg.data.source == "on_the_fly":
        if cfg.data.on_the_fly is None:
            raise ValueError(
                "data.on_the_fly must be defined when "
                "data.source is 'on_the_fly'."
            )

        on_the_fly = cfg.data.on_the_fly

        if not on_the_fly.simulation_config.strip():
            raise ValueError(
                "data.on_the_fly.simulation_config "
                "must not be empty."
            )

        if not on_the_fly.resources.version.strip():
            raise ValueError(
                "data.on_the_fly.resources.version "
                "must not be empty."
            )

        resources_filename = (
            on_the_fly.resources.filename
        )

        if not resources_filename.strip():
            raise ValueError(
                "data.on_the_fly.resources.filename "
                "must not be empty."
            )

        if "{version}" not in resources_filename:
            raise ValueError(
                "data.on_the_fly.resources.filename "
                "must contain '{version}'."
            )

    # ---------------------------------------------------------
    # Precomputed data
    # ---------------------------------------------------------
    if cfg.data.source == "precomputed":
        if cfg.data.precomputed is None:
            raise ValueError(
                "data.precomputed must be defined when "
                "data.source is 'precomputed'."
            )

        precomputed = cfg.data.precomputed

        if not precomputed.version.strip():
            raise ValueError(
                "data.precomputed.version "
                "must not be empty."
            )

        if not precomputed.train_data_filename.strip():
            raise ValueError(
                "data.precomputed.train_data_filename "
                "must not be empty."
            )

        if (
            "{version}"
            not in precomputed.train_data_filename
        ):
            raise ValueError(
                "data.precomputed.train_data_filename "
                "must contain '{version}'."
            )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    if not cfg.output.base_dir.strip():
        raise ValueError(
            "output.base_dir must not be empty."
        )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    if cfg.training.batch_size <= 0:
        raise ValueError(
            "training.batch_size must be > 0."
        )

    if cfg.training.num_workers < 0:
        raise ValueError(
            "training.num_workers must be >= 0."
        )

    if cfg.training.epochs <= 0:
        raise ValueError(
            "training.epochs must be > 0."
        )

    if cfg.data.source == "on_the_fly":
        if cfg.training.n_batches <= 0:
            raise ValueError(
                "training.n_batches must be > 0 for "
                "on-the-fly simulation."
            )

    else:
        if (
            cfg.training.n_batches == 0
            or cfg.training.n_batches < -1
        ):
            raise ValueError(
                "training.n_batches must be -1 or > 0 "
                "for precomputed data."
            )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    if (
        cfg.validation.mode
        not in VALID_VALIDATION_MODES
    ):
        raise ValueError(
            "validation.mode must be one of "
            f"{sorted(VALID_VALIDATION_MODES)}, "
            f"but found {cfg.validation.mode!r}."
        )

    if cfg.validation.seed < 0:
        raise ValueError(
            "validation.seed must be >= 0."
        )

    if cfg.validation.batch_size <= 0:
        raise ValueError(
            "validation.batch_size must be > 0."
        )

    if cfg.validation.mode == "fixed_on_start":
        if cfg.validation.n_spectra <= 0:
            raise ValueError(
                "validation.n_spectra must be > 0 "
                "when validation.mode is "
                "'fixed_on_start'."
            )

    if cfg.validation.mode == "precomputed":
        if (
            cfg.validation.n_spectra == 0
            or cfg.validation.n_spectra < -1
        ):
            raise ValueError(
                "validation.n_spectra must be -1 or > 0 "
                "when validation.mode is 'precomputed'."
            )

    if (
        cfg.data.source == "on_the_fly"
        and cfg.validation.mode != "fixed_on_start"
    ):
        raise ValueError(
            "validation.mode must be 'fixed_on_start' "
            "when data.source is 'on_the_fly'."
        )

    if (
        cfg.data.source == "precomputed"
        and cfg.validation.mode != "precomputed"
    ):
        raise ValueError(
            "validation.mode must be 'precomputed' "
            "when data.source is 'precomputed'."
        )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    if cfg.optim.lr <= 0:
        raise ValueError(
            "optim.lr must be > 0."
        )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------
    if not cfg.scheduler.milestones:
        raise ValueError(
            "scheduler.milestones must not be empty."
        )

    if any(
        milestone <= 0
        for milestone in cfg.scheduler.milestones
    ):
        raise ValueError(
            "All scheduler.milestones must be > 0."
        )

    if cfg.scheduler.milestones != sorted(
        cfg.scheduler.milestones
    ):
        raise ValueError(
            "scheduler.milestones must be sorted "
            "in ascending order."
        )

    if len(cfg.scheduler.milestones) != len(
        set(cfg.scheduler.milestones)
    ):
        raise ValueError(
            "scheduler.milestones contains "
            "duplicate entries."
        )

    if cfg.scheduler.gamma <= 0:
        raise ValueError(
            "scheduler.gamma must be > 0."
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    if (
        cfg.model.architecture
        not in VALID_MODEL_ARCHITECTURES
    ):
        raise ValueError(
            "model.architecture must be one of "
            f"{sorted(VALID_MODEL_ARCHITECTURES)}, "
            f"but found {cfg.model.architecture!r}."
        )

    if cfg.model.n_layers <= 0:
        raise ValueError(
            "model.n_layers must be > 0."
        )

    if cfg.model.n_filters <= 0:
        raise ValueError(
            "model.n_filters must be > 0."
        )

    if cfg.model.in_channels <= 0:
        raise ValueError(
            "model.in_channels must be > 0."
        )

    if cfg.model.out_channels <= 0:
        raise ValueError(
            "model.out_channels must be > 0."
        )

    if not 0 <= cfg.model.dropout < 1:
        raise ValueError(
            "model.dropout must be in [0, 1)."
        )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------
    if (
        cfg.checkpoint.preload
        and not cfg.checkpoint.preload_model.strip()
    ):
        raise ValueError(
            "checkpoint.preload_model must not be empty "
            "when checkpoint.preload is true."
        )


def build_config(
    raw: dict,
    config_dir: Path | None = None,
) -> TrainConfig:
    """
    Build and validate a typed training configuration.

    Filesystem paths are resolved relative to the directory
    containing the training YAML.

    Resource filename templates remain relative because they are
    interpreted inside each subject directory.
    """

    # ---------------------------------------------------------
    # Run
    # ---------------------------------------------------------
    run_raw = raw["run"]

    run = RunCfg(
        name=str(
            run_raw["name"]
        ),
        seed=int(
            run_raw.get(
                "seed",
                42,
            )
        ),
        gpu=int(
            run_raw["gpu"]
        ),
    )

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    data_raw = raw["data"]

    source = str(
        data_raw.get(
            "source",
            "precomputed",
        )
    )

    on_the_fly_raw = data_raw.get(
        "on_the_fly"
    )

    if on_the_fly_raw is None:
        on_the_fly = None
    else:
        resources_raw = on_the_fly_raw[
            "resources"
        ]

        on_the_fly = OnTheFlyCfg(
            simulation_config=_resolve_path(
                str(
                    on_the_fly_raw[
                        "simulation_config"
                    ]
                ),
                config_dir,
            ),
            resources=OnTheFlyResourcesCfg(
                version=str(
                    resources_raw["version"]
                ),
                filename=str(
                    resources_raw["filename"]
                ),
            ),
        )

    precomputed_raw = data_raw.get(
        "precomputed"
    )

    if precomputed_raw is None:
        precomputed = None
    else:
        precomputed = PrecomputedCfg(
            version=str(
                precomputed_raw["version"]
            ),
            train_data_filename=str(
                precomputed_raw.get(
                    "train_data_filename",
                    "TrainData_{version}.h5",
                )
            ),
        )

    data = DataCfg(
        source=source,
        base_dir=_resolve_path(
            str(data_raw["base_dir"]),
            config_dir,
        ),
        train_subjects=[
            str(subject)
            for subject
            in data_raw["train_subjects"]
        ],
        val_subjects=[
            str(subject)
            for subject
            in data_raw["val_subjects"]
        ],
        normalization=str(
            data_raw.get(
                "normalization",
                "projection_energy",
            )
        ),
        on_the_fly=on_the_fly,
        precomputed=precomputed,
    )

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    output_raw = raw["output"]

    output = OutputCfg(
        base_dir=_resolve_path(
            str(output_raw["base_dir"]),
            config_dir,
        ),
        overwrite=bool(
            output_raw.get(
                "overwrite",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    training_raw = raw["training"]

    training = TrainingCfg(
        enabled=bool(
            training_raw.get(
                "enabled",
                True,
            )
        ),
        batch_size=int(
            training_raw["batch_size"]
        ),
        num_workers=int(
            training_raw.get(
                "num_workers",
                0,
            )
        ),
        epochs=int(
            training_raw["epochs"]
        ),
        n_batches=int(
            training_raw.get(
                "n_batches",
                -1,
            )
        ),
        verbose=bool(
            training_raw.get(
                "verbose",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    validation_raw = raw.get(
        "validation",
        {},
    )

    default_validation_mode = (
        "fixed_on_start"
        if source == "on_the_fly"
        else "precomputed"
    )

    default_n_spectra = (
        35000
        if source == "on_the_fly"
        else -1
    )

    validation = ValidationCfg(
        mode=str(
            validation_raw.get(
                "mode",
                default_validation_mode,
            )
        ),
        seed=int(
            validation_raw.get(
                "seed",
                12345,
            )
        ),
        n_spectra=int(
            validation_raw.get(
                "n_spectra",
                default_n_spectra,
            )
        ),
        batch_size=int(
            validation_raw.get(
                "batch_size",
                training.batch_size,
            )
        ),
    )

    # ---------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------
    optim_raw = raw["optim"]

    optim = OptimCfg(
        lr=float(
            optim_raw["lr"]
        ),
    )

    # ---------------------------------------------------------
    # Scheduler
    # ---------------------------------------------------------
    scheduler_raw = raw["scheduler"]

    scheduler = SchedulerCfg(
        milestones=[
            int(milestone)
            for milestone
            in scheduler_raw["milestones"]
        ],
        gamma=float(
            scheduler_raw["gamma"]
        ),
    )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------
    model_raw = raw["model"]

    model = ModelCfg(
        architecture=str(
            model_raw.get(
                "architecture",
                "ynet",
            )
        ),
        n_layers=int(
            model_raw["n_layers"]
        ),
        n_filters=int(
            model_raw["n_filters"]
        ),
        in_channels=int(
            model_raw["in_channels"]
        ),
        out_channels=int(
            model_raw["out_channels"]
        ),
        dropout=float(
            model_raw.get(
                "dropout",
                0.0,
            )
        ),
    )

    # ---------------------------------------------------------
    # Checkpoint
    # ---------------------------------------------------------
    checkpoint_raw = raw.get(
        "checkpoint",
        {},
    )

    preload_model_raw = str(
        checkpoint_raw.get(
            "preload_model",
            "",
        )
    )

    checkpoint = CheckpointCfg(
        preload=bool(
            checkpoint_raw.get(
                "preload",
                False,
            )
        ),
        preload_model=_resolve_path(
            preload_model_raw,
            config_dir,
        ),
    )

    # ---------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------
    prediction_raw = raw.get(
        "prediction",
        {},
    )

    prediction = PredictionCfg(
        enabled=bool(
            prediction_raw.get(
                "enabled",
                False,
            )
        ),
    )

    # ---------------------------------------------------------
    # Complete configuration
    # ---------------------------------------------------------
    cfg = TrainConfig(
        run=run,
        data=data,
        output=output,
        training=training,
        validation=validation,
        optim=optim,
        scheduler=scheduler,
        model=model,
        checkpoint=checkpoint,
        prediction=prediction,
    )

    validate_config(
        cfg
    )

    return cfg