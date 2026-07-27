from types import SimpleNamespace

import pytest
import torch

from walinet.training.training import train_one_epoch
from walinet.training_data.acquisition_length import (
    simulate_acquisition_length,
)


def _simulation_config(
    *,
    zero_filling: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        acquisition=SimpleNamespace(
            n_timepoints=8,
            min_acquired_n_timepoints=4,
            max_acquired_n_timepoints=8,
            zero_filling=zero_filling,
        )
    )


def test_acquisition_length_override_sets_complete_batch_length() -> None:
    spectra = torch.complex(
        torch.randn(3, 8),
        torch.randn(3, 8),
    )

    result = simulate_acquisition_length(
        spectra=spectra,
        config=_simulation_config(zero_filling=False),
        generator=torch.Generator().manual_seed(123),
        acquisition_length_override=8,
    )

    assert result.n_timepoints == 8
    assert torch.equal(
        result.acquired_n_timepoints,
        torch.full((3,), 8, dtype=torch.int64),
    )


def test_acquisition_length_override_must_be_in_configured_range() -> None:
    spectra = torch.complex(
        torch.randn(2, 8),
        torch.randn(2, 8),
    )

    with pytest.raises(
        ValueError,
        match="acquisition_length_override",
    ):
        simulate_acquisition_length(
            spectra=spectra,
            config=_simulation_config(zero_filling=False),
            generator=torch.Generator().manual_seed(123),
            acquisition_length_override=9,
        )


class _RecordingSimulator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.config = _simulation_config(
            zero_filling=False,
        )
        self.overrides: list[int | None] = []

    def simulate(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
        acquisition_length_override: int | None = None,
    ) -> SimpleNamespace:
        del generator
        self.overrides.append(
            acquisition_length_override
        )

        n_timepoints = (
            acquisition_length_override
            if acquisition_length_override is not None
            else 4
        )

        network_input = torch.randn(
            batch_size,
            2,
            n_timepoints,
        )

        return SimpleNamespace(
            network_input=network_input,
            network_target=2 * network_input,
            network_l2=None,
            device=self.device,
            batch_size=batch_size,
            retries_used=0,
        )


def test_first_training_batch_requests_maximum_acquisition_length() -> None:
    simulator = _RecordingSimulator()
    model = torch.nn.Conv1d(
        in_channels=2,
        out_channels=2,
        kernel_size=1,
    )

    train_one_epoch(
        model=model,
        simulator=simulator,
        generator=torch.Generator().manual_seed(123),
        optimizer=torch.optim.SGD(
            model.parameters(),
            lr=0.01,
        ),
        loss_func=torch.nn.MSELoss(),
        architecture="unet",
        batch_size=2,
        n_batches=2,
        verbose=False,
        device=torch.device("cpu"),
        epoch=0,
    )

    assert simulator.overrides == [8, None]
