"""Tests for the isolated NIR capability probe."""

import sys

import pytest

from snn_interpreter.nir_bridge import api

_EXPECTED_KEYS = {
    "nir_available",
    "nirtorch_available",
    "nir_version",
    "nirtorch_version",
    "node_primitives",
}


def test_capability_returns_expected_keys() -> None:
    """The probe reports exactly the documented keys."""
    report = api.capability()
    assert set(report) == _EXPECTED_KEYS
    assert set(report) == set(api.REPORT_KEYS)


def test_node_primitives_are_strings() -> None:
    """Detected primitives are plain names suitable for JSON."""
    primitives = api.capability()["node_primitives"]
    assert isinstance(primitives, list)
    assert all(isinstance(name, str) for name in primitives)


def test_capability_survives_absent_nir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing nir/nirtorch must not raise and stays reported absent."""
    monkeypatch.setitem(sys.modules, "nir", None)
    monkeypatch.setitem(sys.modules, "nirtorch", None)
    report = api.capability()
    assert report["nir_available"] is False
    assert report["nirtorch_available"] is False
    assert report["nir_version"] is None
    assert report["nirtorch_version"] is None
    assert report["node_primitives"] == []


def test_installed_nir_is_detected() -> None:
    """When nir is importable the probe reports a version and primitives."""
    report = api.capability()
    if not report["nir_available"]:
        pytest.skip("nir is not installed")
    assert report["nir_version"]
    assert report["node_primitives"]
