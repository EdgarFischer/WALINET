from walinet.config.build import build_config
from walinet.utils.legacy_params import cfg_to_params


def minimal_raw_config():
    return {
        "run": {
            "name": "test_run",
            "seed": 42,
            "gpu": 0,
        },
        "data": {
            "base_dir": "../data",
            "train_subjects": ["sub_train"],
            "val_subjects": ["sub_val"],
            "version": "v_test",
            "train_data_filename": "TrainData_{version}.h5",
        },
        "output": {
            "base_dir": "../models",
            "overwrite": False,
        },
        "training": {
            "enabled": True,
            "batch_size": 2,
            "num_workers": 0,
            "epochs": 1,
            "n_batches": -1,
            "n_val_batches": -1,
            "verbose": False,
        },
        "optim": {
            "lr": 1e-4,
        },
        "scheduler": {
            "milestones": [10],
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
        "prediction": {
            "enabled": False,
        },
    }


def test_build_config_reads_architecture_and_normalization():
    raw = minimal_raw_config()
    raw["data"]["normalization"] = "max_abs"

    cfg = build_config(raw)

    assert cfg.model.architecture == "unet"
    assert cfg.data.normalization == "max_abs"


def test_build_config_defaults_are_backward_compatible():
    raw = minimal_raw_config()

    raw["data"].pop("normalization", None)
    raw["model"].pop("architecture", None)

    cfg = build_config(raw)

    assert cfg.model.architecture == "ynet"
    assert cfg.data.normalization == "projection_energy"


def test_cfg_to_params_contains_refactor_keys():
    raw = minimal_raw_config()
    raw["data"]["normalization"] = "max_abs"

    cfg = build_config(raw)
    params = cfg_to_params(cfg)

    assert params["architecture"] == "unet"
    assert params["normalization"] == "max_abs"