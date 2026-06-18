import pytest
import torch

from walinet.training.training import forward_model


class DummyYNet:
    def __call__(self, x, y):
        return x + y


class DummyUNet:
    def __call__(self, x):
        return 2 * x


def test_forward_model_dispatches_to_ynet():
    x = torch.ones(1, 2, 8)
    y = 3 * torch.ones(1, 2, 8)

    out = forward_model(
        model=DummyYNet(),
        spectra_all=x,
        spectra_idlip=y,
        params={"architecture": "ynet"},
    )

    assert torch.allclose(out, x + y)


def test_forward_model_dispatches_to_unet():
    x = torch.ones(1, 2, 8)
    y = 3 * torch.ones(1, 2, 8)

    out = forward_model(
        model=DummyUNet(),
        spectra_all=x,
        spectra_idlip=y,
        params={"architecture": "unet"},
    )

    assert torch.allclose(out, 2 * x)


def test_forward_model_defaults_to_ynet_for_legacy_params():
    x = torch.ones(1, 2, 8)
    y = 3 * torch.ones(1, 2, 8)

    out = forward_model(
        model=DummyYNet(),
        spectra_all=x,
        spectra_idlip=y,
        params={},
    )

    assert torch.allclose(out, x + y)


def test_forward_model_rejects_unknown_architecture():
    x = torch.ones(1, 2, 8)
    y = 3 * torch.ones(1, 2, 8)

    with pytest.raises(ValueError):
        forward_model(
            model=DummyUNet(),
            spectra_all=x,
            spectra_idlip=y,
            params={"architecture": "ymodel"},
        )