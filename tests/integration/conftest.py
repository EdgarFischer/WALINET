# tests/integration/conftest.py

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable

import pytest
import torch


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(
        0,
        str(SRC),
    )


@pytest.fixture(scope="session")
def training_config_path() -> Path:
    """
    Return the training configuration used for integration tests.

    A custom configuration can be selected with:

        WALINET_TEST_CONFIG=/path/to/config.yaml pytest

    Otherwise the normal 7T training configuration is used.
    """
    configured_path = os.environ.get(
        "WALINET_TEST_CONFIG",
        str(
            ROOT
            / "configs"
            / "Training"
            / "train_7T.yaml"
        ),
    )

    path = Path(
        configured_path
    ).expanduser()

    if not path.is_absolute():
        path = ROOT / path

    path = path.resolve()

    if not path.is_file():
        raise pytest.UsageError(
            "Integration-test training configuration "
            "does not exist:\n"
            f"  {path}\n\n"
            "Set WALINET_TEST_CONFIG to a valid "
            "training YAML."
        )

    return path


@pytest.fixture(scope="session")
def simulation_system(
    training_config_path: Path,
):
    """
    Build the complete simulation system once for the test session.

    These are GPU integration tests. They are skipped cleanly when
    CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        pytest.skip(
            "WALINET integration tests require CUDA."
        )

    from walinet.training_data.build_simulation_system import (
        build_simulation_system,
    )

    return build_simulation_system(
        training_config_path
    )


@pytest.fixture
def make_generator(
    simulation_system,
) -> Callable[[int], torch.Generator]:
    """
    Return a function that creates a fresh generator on the same
    device as the simulator.
    """

    def factory(
        seed: int,
    ) -> torch.Generator:
        generator = torch.Generator(
            device=simulation_system.device
        )

        generator.manual_seed(
            int(seed)
        )

        return generator

    return factory
