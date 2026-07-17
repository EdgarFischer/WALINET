from pathlib import Path

from walinet.config.build import build_config


def minimal_raw_config(
    tmp_path: Path,
) -> dict:
    return {
        "run": {
            "name": "test_run",
            "seed": 42,
            "gpu": 0,
        },
        "data": {
            "base_dir": "data",
            "train_subjects": [
                "sub_train",
            ],
            "val_subjects": [
                "sub_val",
            ],
            "simulation_config": (
                "simulation.yaml"
            ),
            "resources": {
                "version": "v_test",
                "filename": (
                    "SimulationResources_"
                    "{version}.pt"
                ),
            },
        },
        "output": {
            "base_dir": "models",
            "overwrite": False,
        },
        "training": {
            "batch_size": 2,
            "epochs": 1,
            "n_batches": 1,
            "verbose": False,
        },
        "validation": {
            "seed": 12345,
            "n_spectra": 4,
            "batch_size": 2,
        },
        "optim": {
            "lr": 1e-4,
        },
        "scheduler": {
            "milestones": [
                10,
            ],
            "gamma": 0.5,
        },
        "model": {
            "architecture": "unet",
            "n_layers": 3,
            "n_filters": 8,
            "in_channels": 2,
            "out_channels": 2,
            "dropout": 0.0,
        },
        "checkpoint": {
            "preload": False,
            "preload_model": "",
        },
    }


def test_build_config_reads_current_fields(
    tmp_path: Path,
) -> None:
    raw = minimal_raw_config(
        tmp_path
    )

    cfg = build_config(
        raw,
        config_dir=tmp_path,
    )

    assert cfg.run.name == "test_run"
    assert cfg.run.seed == 42
    assert cfg.run.gpu == 0

    assert cfg.model.architecture == "unet"
    assert cfg.model.n_layers == 3
    assert cfg.model.n_filters == 8
    assert cfg.model.dropout == 0.0

    assert cfg.data.train_subjects == [
        "sub_train",
    ]

    assert cfg.data.val_subjects == [
        "sub_val",
    ]

    assert cfg.data.resources.version == "v_test"

    assert (
        cfg.data.resources.filename
        == "SimulationResources_{version}.pt"
    )

    assert cfg.training.batch_size == 2
    assert cfg.training.epochs == 1
    assert cfg.training.n_batches == 1

    assert cfg.validation.seed == 12345
    assert cfg.validation.n_spectra == 4
    assert cfg.validation.batch_size == 2


def test_build_config_resolves_relative_paths(
    tmp_path: Path,
) -> None:
    raw = minimal_raw_config(
        tmp_path
    )

    cfg = build_config(
        raw,
        config_dir=tmp_path,
    )

    assert Path(
        cfg.data.base_dir
    ) == (
        tmp_path
        / "data"
    ).resolve()

    assert Path(
        cfg.data.simulation_config
    ) == (
        tmp_path
        / "simulation.yaml"
    ).resolve()

    assert Path(
        cfg.output.base_dir
    ) == (
        tmp_path
        / "models"
    ).resolve()


def test_build_config_applies_defaults(
    tmp_path: Path,
) -> None:
    raw = minimal_raw_config(
        tmp_path
    )

    raw["model"].pop(
        "architecture"
    )

    raw["training"].pop(
        "verbose"
    )

    raw["validation"].pop(
        "seed"
    )

    raw["validation"].pop(
        "batch_size"
    )

    raw.pop(
        "checkpoint"
    )

    cfg = build_config(
        raw,
        config_dir=tmp_path,
    )

    assert cfg.model.architecture == "unet"
    assert cfg.training.verbose is False

    assert cfg.validation.seed == 12345

    assert (
        cfg.validation.batch_size
        == cfg.training.batch_size
    )

    assert cfg.checkpoint.preload is False
    assert cfg.checkpoint.preload_model == ""


def test_preload_model_is_resolved_relative_to_output_directory(
    tmp_path: Path,
) -> None:
    raw = minimal_raw_config(
        tmp_path
    )

    raw["checkpoint"] = {
        "preload": True,
        "preload_model": "existing_model",
    }

    cfg = build_config(
        raw,
        config_dir=tmp_path,
    )

    expected_model_directory = (
        tmp_path
        / "models"
        / "existing_model"
    ).resolve()

    assert cfg.checkpoint.preload is True

    assert Path(
        cfg.checkpoint.preload_model
    ) == expected_model_directory