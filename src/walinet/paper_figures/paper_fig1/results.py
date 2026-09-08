"""Read Paper Figure 1 evaluation results."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


def load_results(path: str | Path) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    result = {}
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject, subject_group in h5["subjects"].items():
            result[subject] = {}
            for model, model_group in subject_group["models"].items():
                result[subject][model] = {
                    "nuisance_relative_l2": model_group["nuisance_relative_l2"][:],
                    "metabolite_relative_l2": model_group["metabolite_relative_l2"][:],
                }
    return result


def results_dataframe(path: str | Path):
    """Return compact per-subject/model summary statistics."""
    import pandas as pd

    rows = []
    with h5py.File(Path(path).expanduser(), "r") as h5:
        for subject, subject_group in h5["subjects"].items():
            for model, model_group in subject_group["models"].items():
                family = str(model_group.attrs["family"])
                for metric in ("nuisance_relative_l2", "metabolite_relative_l2"):
                    values = model_group[metric][:]
                    rows.append({
                        "subject": subject,
                        "model": model,
                        "family": family,
                        "metric": metric,
                        "mean": float(np.mean(values)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                    })
    return pd.DataFrame.from_records(rows)
