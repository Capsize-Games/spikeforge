"""Tests for the isolated tonic capability probe."""

import sys

import pytest

from spikeforge.events import tonic_api

_EXPECTED_KEYS = {"tonic_available", "tonic_version", "dataset_classes"}


def test_capability_returns_expected_keys() -> None:
    """The probe reports exactly the documented keys."""
    report = tonic_api.capability()
    assert set(report) == _EXPECTED_KEYS
    assert set(report) == set(tonic_api.REPORT_KEYS)


def test_dataset_classes_are_strings() -> None:
    """Detected dataset classes are plain names suitable for JSON."""
    names = tonic_api.capability()["dataset_classes"]
    assert isinstance(names, list)
    assert all(isinstance(name, str) for name in names)


def test_capability_survives_absent_tonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing tonic must not raise and stays reported absent."""
    monkeypatch.setitem(sys.modules, "tonic", None)
    monkeypatch.setitem(sys.modules, "tonic.datasets", None)
    report = tonic_api.capability()
    assert report["tonic_available"] is False
    assert report["tonic_version"] is None
    assert report["dataset_classes"] == []
    assert tonic_api.available() is False
    assert tonic_api.dataset_class("NMNIST") is None


def test_installed_tonic_is_detected() -> None:
    """When tonic is importable the probe reports a version and classes."""
    report = tonic_api.capability()
    if not report["tonic_available"]:
        pytest.skip("tonic is not installed")
    assert report["tonic_version"]
    assert "NMNIST" in report["dataset_classes"]
    assert tonic_api.dataset_class("NMNIST") is not None
