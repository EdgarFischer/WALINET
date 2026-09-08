"""Configuration loading for the Paper Figure 1 model comparison."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    path: Path
    checkpoint: str


@dataclass(frozen=True)
class EvaluationConfig:
    source_path: Path
    training_config: Path
    subjects: tuple[str, ...]
    models: tuple[ModelSpec, ...]
    n_spectra_per_subject: int
    batch_size: int
    seed: int
    device: str
    output_path: Path


def _resolve(path: str | Path, *, relative_to: Path) -> Path:
    value = Path(path).expanduser()
    return (relative_to / value).resolve() if not value.is_absolute() else value.resolve()


def load_evaluation_config(path: str | Path) -> EvaluationConfig:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    if not isinstance(raw, dict):
        raise TypeError("Evaluation config must contain a YAML mapping.")

    simulation = raw.get("simulation")
    raw_models = raw.get("models")
    output = raw.get("output")
    if not isinstance(simulation, dict) or not isinstance(raw_models, list):
        raise TypeError("Config requires 'simulation' mapping and 'models' list.")
    if not isinstance(output, dict):
        raise TypeError("Config requires an 'output' mapping.")

    subjects = tuple(str(value) for value in simulation.get("subjects", ()))
    if not subjects or len(set(subjects)) != len(subjects):
        raise ValueError("simulation.subjects must be non-empty and unique.")

    models = []
    for entry in raw_models:
        if not isinstance(entry, dict):
            raise TypeError("Every model entry must be a mapping.")
        if not bool(entry.get("enabled", True)):
            continue
        family = str(entry["family"]).lower()
        if family not in {"ynet", "unet", "unet_parameter_matched"}:
            raise ValueError(f"Unknown model family: {family!r}")
        models.append(
            ModelSpec(
                name=str(entry["name"]),
                family=family,
                path=_resolve(entry["path"], relative_to=path.parent),
                checkpoint=str(entry.get("checkpoint", "model_best.pt")),
            )
        )
    if not models or len({model.name for model in models}) != len(models):
        raise ValueError("Enabled model names must be non-empty and unique.")

    n_spectra = int(simulation.get("n_spectra_per_subject", 200_000))
    batch_size = int(simulation.get("batch_size", 1_000))
    if n_spectra <= 0 or batch_size <= 0:
        raise ValueError("Spectrum count and batch size must be positive.")

    return EvaluationConfig(
        source_path=path,
        training_config=_resolve(raw["training_config"], relative_to=path.parent),
        subjects=subjects,
        models=tuple(models),
        n_spectra_per_subject=n_spectra,
        batch_size=batch_size,
        seed=int(simulation.get("seed", 42)),
        device=str(raw.get("device", "cuda:0")),
        output_path=_resolve(output["path"], relative_to=path.parent),
    )
