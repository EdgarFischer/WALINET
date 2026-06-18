import pytest

from walinet.inference.inference import (
    _resolve_architecture,
    _resolve_normalization,
)


def test_resolve_architecture_defaults_to_ynet_for_legacy_params():
    params = {}

    architecture = _resolve_architecture(params, architecture="auto")

    assert architecture == "ynet"


def test_resolve_architecture_reads_unet_from_params():
    params = {"architecture": "unet"}

    architecture = _resolve_architecture(params, architecture="auto")

    assert architecture == "unet"


def test_resolve_architecture_reads_ynet_from_params():
    params = {"architecture": "ynet"}

    architecture = _resolve_architecture(params, architecture="auto")

    assert architecture == "ynet"


def test_resolve_architecture_manual_override_wins():
    params = {"architecture": "unet"}

    architecture = _resolve_architecture(params, architecture="ynet")

    assert architecture == "ynet"


def test_resolve_architecture_rejects_invalid_value():
    params = {"architecture": "ymodel"}

    with pytest.raises(ValueError):
        _resolve_architecture(params, architecture="auto")


def test_resolve_normalization_defaults_to_projection_energy_for_legacy_params():
    params = {}

    normalization = _resolve_normalization(params, normalization="auto")

    assert normalization == "projection_energy"


def test_resolve_normalization_reads_max_abs_from_params():
    params = {"normalization": "max_abs"}

    normalization = _resolve_normalization(params, normalization="auto")

    assert normalization == "max_abs"


def test_resolve_normalization_reads_projection_energy_from_params():
    params = {"normalization": "projection_energy"}

    normalization = _resolve_normalization(params, normalization="auto")

    assert normalization == "projection_energy"


def test_resolve_normalization_manual_override_wins():
    params = {"normalization": "max_abs"}

    normalization = _resolve_normalization(
        params,
        normalization="projection_energy",
    )

    assert normalization == "projection_energy"


def test_resolve_normalization_rejects_invalid_value():
    params = {"normalization": "energy"}

    with pytest.raises(ValueError):
        _resolve_normalization(params, normalization="auto")