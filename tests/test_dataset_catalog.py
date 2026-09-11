"""Registry modality metadata and catalog back-compatibility."""

import json

from snn_interpreter.data import datasets
from snn_interpreter.data.dataset_spec import DatasetSpec


def test_catalog_includes_modality_and_availability() -> None:
    """Every catalog entry carries a modality and an availability flag."""
    entries = datasets.catalog()
    assert entries
    for entry in entries:
        assert entry["modality"] in {"image", "event", "sequence"}
        assert isinstance(entry["available"], bool)
        assert set(entry) == {
            "name", "classes", "description", "modality", "available",
        }


def test_catalog_preserves_existing_image_metadata() -> None:
    """Image datasets keep their historic names, classes, and text."""
    by_name = {entry["name"]: entry for entry in datasets.catalog()}
    assert set(by_name) == set(datasets.dataset_names())
    assert by_name["mnist"]["classes"] == 10
    assert by_name["mnist"]["description"] == "Handwritten digits (0-9)"
    assert by_name["emnist_letters"]["classes"] == 26
    assert by_name["cifar10"]["classes"] == 10
    for entry in by_name.values():
        if entry["modality"] == "image":
            assert entry["available"] is True


def test_dataset_info_is_unchanged() -> None:
    """``dataset_info`` keeps its ``(num_classes, description)`` contract."""
    assert datasets.dataset_info("mnist") == (10, "Handwritten digits (0-9)")
    unknown = datasets.dataset_info("does-not-exist")
    assert unknown == (10, "Handwritten digits (0-9)")


def test_dataset_spec_carries_name_and_modality() -> None:
    """The resolved spec exposes its name and modality."""
    spec = datasets.dataset_spec("kmnist")
    assert isinstance(spec, DatasetSpec)
    assert spec.name == "kmnist"
    assert spec.modality == "image"
    assert datasets.dataset_modality("kmnist") == "image"


def test_dataset_spec_defaults_to_image_modality() -> None:
    """A spec without an explicit modality is an image dataset."""
    spec = DatasetSpec("synthetic", 2, "Synthetic stream")
    assert spec.modality == "image"
    assert spec.cls is None
    assert spec.kwargs == {}
    assert spec.tonic_class is None


def test_catalog_is_json_serialisable() -> None:
    """The catalog survives ``json.dumps`` for the WebSocket payload."""
    assert isinstance(json.dumps(datasets.catalog()), str)
