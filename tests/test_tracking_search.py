"""Search/filter behaviour over the file-based checkpoint registry."""

from typing import Any, Dict, List, Optional

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.network.model_search import search_models


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _save(
    name: str,
    dataset: str,
    topology: str,
    coding: str,
    device: str,
    accuracy: Optional[float],
) -> None:
    """Write one checkpoint with the given searchable metadata."""
    meta = {
        "dataset": dataset,
        "topology": topology,
        "coding": coding,
        "input_mode": coding,
        "device": device,
    }
    history = (
        [] if accuracy is None else [{"step": 1, "test_accuracy": accuracy}]
    )
    model_store.save(name, torch.nn.Linear(4, 2), meta, history)


def _names(records: List[Dict[str, Any]]) -> List[str]:
    """Return the sorted names carried by the records."""
    return sorted(record["name"] for record in records)


@pytest.fixture
def _registry() -> None:
    """Populate a small mixed registry, including a legacy entry."""
    _save("mnist_rate", "mnist", "fc_legacy", "rate", "cpu", 80.0)
    _save("mnist_latency", "mnist", "fc_small", "latency", "cpu", 60.0)
    _save("cifar_rate", "cifar10", "conv_net", "rate", "cuda", 95.0)
    _save("legacy_entry", "mnist", "fc_legacy", "raw", "cpu", None)


def test_filter_by_dataset(_registry: None) -> None:
    """A dataset filter keeps only the matching checkpoints."""
    assert _names(search_models(dataset="cifar10")) == ["cifar_rate"]


def test_filter_by_topology_and_coding(_registry: None) -> None:
    """Topology and coding filters select the expected checkpoints."""
    assert _names(search_models(topology="fc_small")) == ["mnist_latency"]
    assert _names(search_models(coding="rate")) == [
        "cifar_rate", "mnist_rate"
    ]


def test_filter_by_device_and_min_accuracy(_registry: None) -> None:
    """Device and minimum-accuracy filters compose."""
    assert _names(search_models(device="cuda")) == ["cifar_rate"]
    assert _names(search_models(min_accuracy=80.0)) == [
        "cifar_rate", "mnist_rate"
    ]


def test_legacy_entry_has_no_accuracy(_registry: None) -> None:
    """A legacy checkpoint stays searchable but reports no accuracy."""
    records = {record["name"]: record for record in search_models()}
    assert records["legacy_entry"]["accuracy"] is None
    assert records["legacy_entry"]["manifest"] is None


def test_filter_by_name_substring(_registry: None) -> None:
    """The name filter is a case-insensitive substring match."""
    assert _names(search_models(name="MNIST")) == [
        "mnist_latency", "mnist_rate"
    ]
