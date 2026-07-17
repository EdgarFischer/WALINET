import pytest
import torch

from walinet.training.training import (
    forward_model,
)


class DummyYNet(torch.nn.Module):
    def forward(
        self,
        network_input: torch.Tensor,
        network_l2: torch.Tensor,
    ) -> torch.Tensor:
        return network_input + network_l2


class DummyUNet(torch.nn.Module):
    def forward(
        self,
        network_input: torch.Tensor,
    ) -> torch.Tensor:
        return 2 * network_input


def test_forward_model_dispatches_to_ynet() -> None:
    network_input = torch.ones(
        1,
        2,
        8,
    )

    network_l2 = 3 * torch.ones(
        1,
        2,
        8,
    )

    output = forward_model(
        model=DummyYNet(),
        network_input=network_input,
        network_l2=network_l2,
        architecture="ynet",
    )

    torch.testing.assert_close(
        output,
        network_input + network_l2,
    )


def test_forward_model_dispatches_to_unet() -> None:
    network_input = torch.ones(
        1,
        2,
        8,
    )

    network_l2 = 3 * torch.ones(
        1,
        2,
        8,
    )

    output = forward_model(
        model=DummyUNet(),
        network_input=network_input,
        network_l2=network_l2,
        architecture="unet",
    )

    torch.testing.assert_close(
        output,
        2 * network_input,
    )


def test_forward_model_ynet_requires_l2_input() -> None:
    network_input = torch.ones(
        1,
        2,
        8,
    )

    with pytest.raises(
        RuntimeError,
        match="YNet requires network_l2",
    ):
        forward_model(
            model=DummyYNet(),
            network_input=network_input,
            network_l2=None,
            architecture="ynet",
        )


def test_forward_model_rejects_unknown_architecture() -> None:
    network_input = torch.ones(
        1,
        2,
        8,
    )

    with pytest.raises(
        ValueError,
        match="Unknown architecture",
    ):
        forward_model(
            model=DummyUNet(),
            network_input=network_input,
            network_l2=None,
            architecture="ymodel",
        )