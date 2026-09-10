"""Manifest round-trip, config hashing, and the legacy fallback."""

from typing import Any

import pytest

from snn_interpreter.network import model_store
from snn_interpreter.network.spiking_net import SpikingNet
from snn_interpreter.tracking.config_hash import config_hash
from snn_interpreter.tracking.manifest import ReproducibilityManifest
from snn_interpreter.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: dict = {"dataset": "mnist", "num_steps": 4, "device": "cpu"}
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_manifest_round_trips_through_checkpoint() -> None:
    """A saved engine records a manifest whose hash matches its config."""
    engine = _engine(seed=7, epochs=2, subset=3, lr=0.02)
    engine.save("tracked", history=[{"step": 1, "test_accuracy": 55.0}])
    stored = model_store.manifest("tracked")
    assert stored["available"] is True
    assert stored["schema_version"] == 1
    assert stored["seed"] == 7
    assert stored["config"]["dataset"] == "mnist"
    assert stored["config"]["topology"] == "fc_legacy"
    assert stored["config"]["epochs"] == 2
    assert stored["config"]["spec"]["stages"]
    assert set(stored["versions"]) >= {"python", "torch", "nir"}
    assert stored["reproducible"]["bit_exact"] is False
    rebuilt = ReproducibilityManifest.from_dict(stored)
    assert rebuilt.config_hash == stored["config_hash"]


def test_config_hash_is_stable_and_key_order_independent() -> None:
    """Equivalent configs hash equally; a changed value changes the hash."""
    left = {"dataset": "mnist", "hidden": 8}
    right = {"hidden": 8, "dataset": "mnist"}
    assert config_hash(left) == config_hash(right)
    assert config_hash(left) != config_hash({"dataset": "mnist", "hidden": 9})


def test_legacy_checkpoint_reports_safe_fallback() -> None:
    """A checkpoint without a manifest loads and reports an explicit gap."""
    net = SpikingNet(hidden=8, beta=0.5, num_classes=10)
    model_store.save("legacy", net, {"dataset": "mnist"})
    stored = model_store.manifest("legacy")
    assert stored["available"] is False
    assert "legacy" in stored["reason"]
    assert stored["meta"]["dataset"] == "mnist"
    assert model_store.load("legacy")["state_dict"]


def test_missing_checkpoint_reports_unavailable() -> None:
    """Reading a manifest for a missing file never raises."""
    stored = model_store.manifest("does_not_exist")
    assert stored["available"] is False
    assert "unavailable" in stored["reason"]
