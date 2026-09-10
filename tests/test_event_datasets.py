"""Event dataset registry entries, availability, and catalog reporting."""

import pytest

from snn_interpreter.data import datasets
from snn_interpreter.events import tonic_api

_EVENT_KEYS = {
    "n_mnist": 10,
    "dvs128_gesture": 11,
    "cifar10_dvs": 10,
    "ssc": 35,
}


def test_event_datasets_are_registered() -> None:
    """Every event dataset is present with modality and class count."""
    for name, classes in _EVENT_KEYS.items():
        spec = datasets.dataset_spec(name)
        assert spec.modality == "event"
        assert spec.num_classes == classes
        assert spec.description
        assert spec.tonic_class
        assert spec.cls is None


def test_event_catalog_entries_report_event_modality() -> None:
    """The catalog marks event datasets and reports availability."""
    by_name = {entry["name"]: entry for entry in datasets.catalog()}
    for name in _EVENT_KEYS:
        entry = by_name[name]
        assert entry["modality"] == "event"
        assert entry["available"] is tonic_api.available()


def test_event_availability_tracks_tonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Availability follows the probe rather than being hard-coded."""
    monkeypatch.setattr(tonic_api, "available", lambda: False)
    assert datasets.dataset_available("n_mnist") is False
    monkeypatch.setattr(tonic_api, "available", lambda: True)
    assert datasets.dataset_available("n_mnist") is True


def test_image_datasets_ignore_tonic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Image availability never depends on the optional tonic package."""
    monkeypatch.setattr(tonic_api, "available", lambda: False)
    assert datasets.dataset_available("mnist") is True
