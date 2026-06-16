# src/walinet/utils/legacy_params.py

from pathlib import Path

def cfg_to_params(cfg) -> dict:
    """
    Convert new YAML-based TrainConfig to the legacy params dictionary.

    This keeps old training/dataloader/initialize code usable during refactor.
    """
    path_to_model = Path(cfg.output.base_dir) / cfg.run.name
    path_to_data = Path(cfg.data.base_dir)

    params = {}

    # Run / paths
    params["model_name"] = cfg.run.name
    params["path_to_model"] = str(path_to_model) + "/"
    params["path_to_data"] = str(path_to_data) + "/"

    # Data
    params["train_subjects"] = list(cfg.data.train_subjects)
    params["val_subjects"] = list(cfg.data.val_subjects)
    params["data_version"] = cfg.data.version

    # Train params
    params["gpu"] = cfg.run.gpu
    params["batch_size"] = cfg.training.batch_size
    params["num_worker"] = cfg.training.num_workers
    params["lr"] = cfg.optim.lr
    params["epochs"] = cfg.training.epochs
    params["verbose"] = cfg.training.verbose
    params["n_batches"] = cfg.training.n_batches
    params["n_val_batches"] = cfg.training.n_val_batches

    # LR scheduler
    params["milestones"] = list(cfg.scheduler.milestones)
    params["gamma"] = cfg.scheduler.gamma

    # Model params
    params["nLayers"] = cfg.model.n_layers
    params["nFilters"] = cfg.model.n_filters
    params["in_channels"] = cfg.model.in_channels
    params["out_channels"] = cfg.model.out_channels
    params["dropout"] = cfg.model.dropout

    # Old control flags
    params["clean_model"] = cfg.output.overwrite
    params["train"] = cfg.training.enabled
    params["predict"] = cfg.prediction.enabled

    # Preload / checkpoint
    params["preload"] = cfg.checkpoint.preload
    params["preload_model"] = cfg.checkpoint.preload_model

    return params