"""Metadata diff classification and checkpoint-level diffing."""

import json
from typing import Any, Dict

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.network.model_diff import checkpoint_diff, metadata_diff


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _save(name: str, meta: Dict[str, Any], config_hash: str) -> None:
    """Write a checkpoint carrying a minimal manifest."""
    manifest = {"config_hash": config_hash, "config": meta}
    model_store.save(name, torch.nn.Linear(4, 2), meta, [], manifest)


def test_metadata_diff_classifies_every_key() -> None:
    """Added, removed, changed, and same keys are classified correctly."""
    diff = metadata_diff({"a": 1, "b": 2}, {"b": 3, "c": 4})
    assert diff["changed"] == ["b"]
    assert diff["added"] == ["c"]
    assert diff["removed"] == ["a"]
    assert diff["identical"] is False
    statuses = {item["key"]: item["status"] for item in diff["entries"]}
    assert statuses == {"a": "removed", "b": "changed", "c": "added"}
    assert json.loads(json.dumps(diff))["changed"] == ["b"]


def test_metadata_diff_identical() -> None:
    """Equal mappings report identical with a single ``same`` entry."""
    diff = metadata_diff({"a": 1}, {"a": 1})
    assert diff["identical"] is True
    assert diff["changed"] == []
    assert diff["entries"][0]["status"] == "same"


def test_checkpoint_diff_matches_config_hash() -> None:
    """Two identical configs read as identical with a matching hash."""
    _save("left", {"dataset": "mnist", "hidden": 8}, "same-hash")
    _save("right", {"dataset": "mnist", "hidden": 8}, "same-hash")
    diff = checkpoint_diff("left", "right")
    assert diff["identical"] is True
    assert diff["config_hash"]["matches"] is True
    assert json.dumps(diff)


def test_checkpoint_diff_flags_missing_checkpoint() -> None:
    """A missing checkpoint raises ``FileNotFoundError`` for the caller."""
    _save("only", {"dataset": "mnist"}, "h")
    with pytest.raises(FileNotFoundError):
        checkpoint_diff("only", "absent")
