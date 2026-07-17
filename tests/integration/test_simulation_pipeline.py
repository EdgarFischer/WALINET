# tests/integration/test_simulation_pipeline.py

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.integration


def _assert_finite(
    tensor: torch.Tensor,
    *,
    name: str,
) -> None:
    assert bool(
        torch.isfinite(
            tensor
        ).all()
    ), f"{name} contains non-finite values."


def _assert_batches_close(
    batch_a,
    batch_b,
) -> None:
    """
    Compare the trainer-relevant outputs of two simulated batches.
    """
    torch.testing.assert_close(
        batch_a.normalized_input_spectra,
        batch_b.normalized_input_spectra,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        batch_a.normalized_target_spectra,
        batch_b.normalized_target_spectra,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        batch_a.network_input,
        batch_b.network_input,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        batch_a.network_target,
        batch_b.network_target,
        rtol=1e-6,
        atol=1e-7,
    )

    torch.testing.assert_close(
        batch_a.normalization_scale,
        batch_b.normalization_scale,
        rtol=1e-6,
        atol=1e-7,
    )

    assert (
        batch_a.network_l2 is None
    ) == (
        batch_b.network_l2 is None
    )

    if batch_a.network_l2 is not None:
        assert batch_b.network_l2 is not None

        torch.testing.assert_close(
            batch_a.network_l2,
            batch_b.network_l2,
            rtol=1e-6,
            atol=1e-7,
        )


def test_build_complete_simulation_system(
    simulation_system,
    training_config_path: Path,
) -> None:
    """
    Integration test:

        training YAML
        -> simulation YAML
        -> basis
        -> resources
        -> train and validation simulators
    """
    system = simulation_system

    assert (
        system.train_config_path
        == training_config_path.resolve()
    )

    assert system.simulation_config_path.is_file()

    assert system.prepared_basis is not None
    assert system.resources is not None
    assert system.metabolite_simulator is not None

    assert system.train_simulator is not None
    assert system.validation_simulator is not None

    assert system.device.type == "cuda"

    expected_n_timepoints = int(
        system
        .simulation_config
        .acquisition
        .n_timepoints
    )

    assert (
        system.train_simulator.n_timepoints
        == expected_n_timepoints
    )

    assert (
        system.validation_simulator.n_timepoints
        == expected_n_timepoints
    )

    assert (
        system.train_simulator.device
        == system.device
    )

    assert (
        system.validation_simulator.device
        == system.device
    )


def test_train_and_validation_use_separate_pools(
    simulation_system,
) -> None:
    """
    Ensure that the train and validation simulators are connected
    to different resource pools and that their configured subject
    lists are disjoint.
    """
    system = simulation_system

    assert (
        system.train_simulator.pool
        is system.resources.train
    )

    assert (
        system.validation_simulator.pool
        is system.resources.validation
    )

    assert (
        system.train_simulator.pool
        is not system.validation_simulator.pool
    )

    train_subjects = set(
        system.train_config.data.train_subjects
    )

    validation_subjects = set(
        system.train_config.data.val_subjects
    )

    assert train_subjects
    assert validation_subjects

    assert train_subjects.isdisjoint(
        validation_subjects
    )


def test_simulated_batch_fulfils_network_contract(
    simulation_system,
    make_generator,
) -> None:
    """
    Simulate one complete batch and verify the public contract
    required by the trainer.
    """
    simulator = (
        simulation_system.train_simulator
    )

    batch_size = 4
    n_timepoints = simulator.n_timepoints

    batch = simulator.simulate(
        batch_size=batch_size,
        generator=make_generator(1001),
    )

    assert batch.batch_size == batch_size
    assert batch.n_timepoints == n_timepoints
    assert batch.device == simulation_system.device

    expected_complex_shape = (
        batch_size,
        n_timepoints,
    )

    expected_network_shape = (
        batch_size,
        2,
        n_timepoints,
    )

    assert (
        batch.normalized_input_spectra.shape
        == expected_complex_shape
    )

    assert (
        batch.normalized_target_spectra.shape
        == expected_complex_shape
    )

    assert (
        batch.network_input.shape
        == expected_network_shape
    )

    assert (
        batch.network_target.shape
        == expected_network_shape
    )

    assert (
        batch.normalization_scale.shape
        == (
            batch_size,
            1,
        )
    )

    assert torch.is_complex(
        batch.normalized_input_spectra
    )

    assert torch.is_complex(
        batch.normalized_target_spectra
    )

    assert not torch.is_complex(
        batch.network_input
    )

    assert not torch.is_complex(
        batch.network_target
    )

    assert torch.is_floating_point(
        batch.network_input
    )

    assert torch.is_floating_point(
        batch.network_target
    )

    _assert_finite(
        batch.network_input,
        name="network_input",
    )

    _assert_finite(
        batch.network_target,
        name="network_target",
    )

    _assert_finite(
        batch.normalization_scale,
        name="normalization_scale",
    )

    assert bool(
        torch.all(
            batch.normalization_scale
            > 0
        )
    )

    # The maximum absolute value of every normalized input
    # spectrum must be one.
    normalized_maximum = torch.amax(
        torch.abs(
            batch.normalized_input_spectra
        ),
        dim=-1,
    )

    torch.testing.assert_close(
        normalized_maximum,
        torch.ones_like(
            normalized_maximum
        ),
        rtol=1e-5,
        atol=1e-6,
    )

    # The real/imaginary network channels must represent exactly
    # the normalized complex input.
    reconstructed_input = torch.complex(
        batch.network_input[:, 0, :],
        batch.network_input[:, 1, :],
    )

    torch.testing.assert_close(
        reconstructed_input,
        batch.normalized_input_spectra,
        rtol=1e-6,
        atol=1e-7,
    )

    if batch.network_l2 is None:
        assert batch.normalized_l2_spectra is None
    else:
        assert (
            batch.network_l2.shape
            == expected_network_shape
        )

        assert (
            batch.normalized_l2_spectra
            is not None
        )

        _assert_finite(
            batch.network_l2,
            name="network_l2",
        )


def test_simulation_is_reproducible_with_fixed_seed(
    simulation_system,
    make_generator,
) -> None:
    """
    Two fresh generators with the same seed must produce the same
    simulated trainer batch.
    """
    simulator = (
        simulation_system.train_simulator
    )

    batch_a = simulator.simulate(
        batch_size=4,
        generator=make_generator(12345),
    )

    batch_b = simulator.simulate(
        batch_size=4,
        generator=make_generator(12345),
    )

    _assert_batches_close(
        batch_a,
        batch_b,
    )

    batch_c = simulator.simulate(
        batch_size=4,
        generator=make_generator(12346),
    )

    assert not torch.allclose(
        batch_a.network_input,
        batch_c.network_input,
        rtol=1e-6,
        atol=1e-7,
    )


def test_fixed_validation_batches_are_created_once_on_cpu(
    simulation_system,
    make_generator,
) -> None:
    """
    Verify the complete fixed-validation workflow, including the
    smaller final batch and transfer from GPU to CPU.
    """
    from walinet.training.training import (
        create_fixed_validation_batches,
    )

    architecture = (
        simulation_system
        .train_config
        .model
        .architecture
    )

    validation_batches = (
        create_fixed_validation_batches(
            simulator=(
                simulation_system
                .validation_simulator
            ),
            generator=make_generator(2001),
            n_spectra=5,
            batch_size=3,
            architecture=architecture,
            verbose=False,
        )
    )

    assert isinstance(
        validation_batches,
        tuple,
    )

    assert len(
        validation_batches
    ) == 2

    assert [
        batch.batch_size
        for batch in validation_batches
    ] == [
        3,
        2,
    ]

    assert sum(
        batch.batch_size
        for batch in validation_batches
    ) == 5

    expected_n_timepoints = (
        simulation_system
        .validation_simulator
        .n_timepoints
    )

    for batch in validation_batches:
        assert (
            batch.n_timepoints
            == expected_n_timepoints
        )

        assert (
            batch.network_input.device.type
            == "cpu"
        )

        assert (
            batch.network_target.device.type
            == "cpu"
        )

        _assert_finite(
            batch.network_input,
            name="fixed network_input",
        )

        _assert_finite(
            batch.network_target,
            name="fixed network_target",
        )

        if batch.network_l2 is not None:
            assert (
                batch.network_l2.device.type
                == "cpu"
            )

            _assert_finite(
                batch.network_l2,
                name="fixed network_l2",
            )


def test_single_training_and_validation_epoch(
    simulation_system,
    make_generator,
) -> None:
    """
    Run the complete trainer-relevant pipeline:

        simulate
        -> network forward
        -> loss
        -> backward
        -> optimizer update
        -> fixed validation
    """
    from torch import nn
    from torch.optim import Adam

    from walinet.model.model import (
        uModel,
        yModel,
    )
    from walinet.training.training import (
        create_fixed_validation_batches,
        train_one_epoch,
        validate_one_epoch,
    )

    system = simulation_system
    cfg = system.train_config

    architecture = (
        cfg.model.architecture
        .strip()
        .lower()
    )

    model_arguments = {
        "nLayers": cfg.model.n_layers,
        "nFilters": cfg.model.n_filters,
        "dropout": cfg.model.dropout,
        "in_channels": cfg.model.in_channels,
        "out_channels": cfg.model.out_channels,
    }

    if architecture == "unet":
        model = uModel(
            **model_arguments
        )

    elif architecture == "ynet":
        model = yModel(
            **model_arguments
        )

    else:
        raise AssertionError(
            "Unexpected architecture in validated config: "
            f"{architecture!r}"
        )

    model = model.to(
        system.device
    )

    optimizer = Adam(
        model.parameters(),
        lr=cfg.optim.lr,
    )

    loss_func = nn.MSELoss()

    first_parameter = next(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    parameter_before = (
        first_parameter
        .detach()
        .clone()
    )

    validation_batches = (
        create_fixed_validation_batches(
            simulator=(
                system.validation_simulator
            ),
            generator=make_generator(3001),
            n_spectra=4,
            batch_size=2,
            architecture=architecture,
            verbose=False,
        )
    )

    model, training_loss = train_one_epoch(
        model=model,
        simulator=system.train_simulator,
        generator=make_generator(3002),
        optimizer=optimizer,
        loss_func=loss_func,
        architecture=architecture,
        batch_size=2,
        n_batches=1,
        verbose=False,
        device=system.device,
        epoch=0,
    )

    assert math.isfinite(
        training_loss
    )

    assert first_parameter.grad is not None

    assert bool(
        torch.isfinite(
            first_parameter.grad
        ).all()
    )

    assert bool(
        torch.any(
            first_parameter.grad
            != 0
        )
    )

    assert not torch.equal(
        parameter_before,
        first_parameter.detach(),
    )

    validation_loss = validate_one_epoch(
        model=model,
        validation_batches=validation_batches,
        loss_func=loss_func,
        architecture=architecture,
        verbose=False,
        device=system.device,
        epoch=0,
    )

    assert math.isfinite(
        validation_loss
    )
