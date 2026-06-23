#!/usr/bin/env python3

from pathlib import Path
import sys
import os
import argparse
import shutil
import random
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "train.yaml"),
        help="Path to training config YAML.",
    )
    return p.parse_args()


def prepare_model_folder(model_dir: Path, config_path: Path, overwrite: bool) -> None:
    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Model directory already exists:\n"
                f"  {model_dir}\n\n"
                f"Choose a different run.name or set output.overwrite: true."
            )

        shutil.rmtree(model_dir)

    (model_dir / "predictions").mkdir(parents=True, exist_ok=True)
    (model_dir / "configs").mkdir(parents=True, exist_ok=True)

    shutil.copy(config_path, model_dir / "configs" / config_path.name)


def write_params_txt(model_dir: Path, params: dict) -> None:
    with open(model_dir / "params.txt", "w") as f:
        for key, val in params.items():
            f.write(f"{key}: {val}\n")


if __name__ == "__main__":
    args = parse_args()

    sys.path.insert(0, str(SRC))
    os.chdir(ROOT)

    from walinet.config.load import load_yaml
    from walinet.config.build import build_config
    from walinet.utils.legacy_params import cfg_to_params

    config_path = Path(args.config).resolve()
    config_dir = config_path.parent

    raw = load_yaml(config_path)
    cfg = build_config(raw, config_dir=config_dir)

    # Important: must happen before importing torch.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.run.gpu)

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from torch.optim.lr_scheduler import MultiStepLR

    from walinet.training.training import training, validation
    from walinet.model.model import yModel, uModel
    from walinet.data.dataloader import SpectrumDatasetLoad

    np.random.seed(cfg.run.seed)
    random.seed(cfg.run.seed)
    torch.manual_seed(cfg.run.seed)

    params = cfg_to_params(cfg)

    model_dir = Path(params["path_to_model"])
    prepare_model_folder(
        model_dir=model_dir,
        config_path=config_path,
        overwrite=cfg.output.overwrite,
    )
    write_params_txt(model_dir, params)

    print(params["model_name"])
    print(f"Model dir: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_dataset = SpectrumDatasetLoad(
        params=params,
        files=params["train_subjects"],
        version=params["data_version"],
        aug=True,
    )

    val_dataset = SpectrumDatasetLoad(
        params=params,
        files=params["val_subjects"],
        version=params["data_version"],
        aug=False,
    )

    train_dataloader = DataLoader(
        train_dataset,
        num_workers=params["num_worker"],
        shuffle=True,
        batch_size=params["batch_size"],
    )

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=params["num_worker"],
        shuffle=True,
        batch_size=params["batch_size"],
    )

    architecture = params.get("architecture", "ynet")

    if architecture == "ynet":
        model = yModel(
            nLayers=params["nLayers"],
            nFilters=params["nFilters"],
            dropout=params["dropout"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
        ).to(device)

    elif architecture == "unet":
        model = uModel(
            nLayers=params["nLayers"],
            nFilters=params["nFilters"],
            dropout=params["dropout"],
            in_channels=params["in_channels"],
            out_channels=params["out_channels"],
        ).to(device)

    else:
        raise ValueError(
            f"Unknown architecture '{architecture}'. "
            "Use 'ynet' or 'unet'."
        )

    print(f"Using architecture: {architecture}")

    if params["preload"]:
        preload_path = (
            Path(cfg.output.base_dir)
            / params["preload_model"]
            / "model_last.pt"
        )

        print(f"Loading model: {preload_path}")
        model.load_state_dict(
            torch.load(preload_path, map_location=device)
        )

    # Runtime objects.
    # For now these are still inserted into a runtime params dict,
    # because training.py expects them there.
    runtime_params = dict(params)
    runtime_params["loss_func"] = nn.MSELoss()
    runtime_params["optimizer"] = Adam(
        model.parameters(),
        lr=params["lr"],
    )
    runtime_params["scheduler"] = MultiStepLR(
        runtime_params["optimizer"],
        milestones=params["milestones"],
        gamma=params["gamma"],
    )

    if params["train"]:
        best_loss = None
        loss_path = model_dir / "loss.txt"

        with open(loss_path, "w") as f:
            f.write("Epoch; Epoch Loss; Validation Loss; Learning Rate;\n")

        for epoch in range(params["epochs"]):
            model, train_loss = training(
                model=model,
                params=runtime_params,
                dataloader=train_dataloader,
                device=device,
                epoch=epoch,
            )

            val_loss = validation(
                model=model,
                params=runtime_params,
                dataloader=val_dataloader,
                device=device,
                epoch=epoch,
            )

            lr = runtime_params["scheduler"].get_last_lr()[0]

            with open(loss_path, "a") as f:
                log = "Epoch: {:03d}, Loss: {:.10f}, Val Loss: {:.10f}, LR: {:.10f}"
                f.write(log.format(epoch + 1, train_loss, val_loss, lr))

                torch.save(
                    model.state_dict(),
                    model_dir / "model_last.pt",
                )

                if best_loss is None or val_loss < best_loss:
                    best_loss = val_loss
                    torch.save(
                        model.state_dict(),
                        model_dir / "model_best.pt",
                    )
                    f.write(", best model")

                f.write("\n")

            runtime_params["scheduler"].step()

    if params["predict"]:
        raise NotImplementedError(
            "Prediction is legacy code and should be moved to a separate script."
        )

    print("All done!")