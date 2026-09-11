"""External tracking sinks: honest degradation and stub delivery."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterator

import pytest

from snn_interpreter.network import model_store
from snn_interpreter.tracking import sink_probe, sinks
from snn_interpreter.tracking.tensorboard_sink import TensorBoardSink
from snn_interpreter.tracking.wandb_sink import WandBSink
from snn_interpreter.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    """Keep every checkpoint write inside the test's temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))
    yield


def _record(requested: str = "tensorboard") -> Dict[str, Any]:
    """Return a manifest-like record with a handful of numeric fields."""
    return {
        "schema_version": 1,
        "seed": 3,
        "config_hash": "abc",
        "history": [{"step": 1, "test_accuracy": 55.0}],
        "tracking": {"requested": requested, "active": False},
    }


def _writer_probe(store: Dict[str, Any]) -> Any:
    """Return a stand-in SummaryWriter class recording scalar writes."""
    def make(log_dir: str = "") -> Any:
        store["log_dir"] = log_dir
        store["scalars"] = []

        def add_scalar(key: str, value: float, step: int) -> None:
            store["scalars"].append((key, value, step))

        def close() -> None:
            store["closed"] = True

        return SimpleNamespace(add_scalar=add_scalar, close=close)

    return make


def _wandb_probe(store: Dict[str, Any]) -> Any:
    """Return a stand-in wandb module recording run lifecycle calls."""
    def init(project: str = "", config: Any = None,
             reinit: bool = False) -> None:
        store["init"] = {"project": project, "config": config}

    def log(payload: Dict[str, Any]) -> None:
        store["logged"] = payload

    def finish() -> None:
        store["finished"] = True

    return SimpleNamespace(init=init, log=log, finish=finish)


def test_no_sink_requested_is_an_honest_noop() -> None:
    """An unconfigured run reports that no sink was asked for."""
    block = sinks.describe(None)
    assert block == {
        "requested": None,
        "active": False,
        "reason": "no sink requested",
    }
    assert sinks.emit({}) is False


def test_unknown_sink_is_a_named_noop() -> None:
    """An unknown sink name degrades to a named no-op, not an error."""
    block = sinks.describe("does-not-exist")
    assert block["active"] is False
    assert "unknown sink" in block["reason"]
    assert sinks.resolve("does-not-exist").name == "does-not-exist"


def test_absent_backend_is_a_named_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing extra becomes a recorded reason and no delivery."""
    monkeypatch.setattr(sink_probe, "tensorboard_available", lambda: False)
    monkeypatch.setattr(sink_probe, "tensorboard_writer", lambda: None)
    block = sinks.describe("tensorboard")
    assert block["active"] is False
    assert "not installed" in block["reason"]
    assert sinks.emit(_record()) is False


def test_tensorboard_stub_receives_scalars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the backend exists, scalars are forwarded to it."""
    store: Dict[str, Any] = {}
    monkeypatch.setattr(sink_probe, "tensorboard_available", lambda: True)
    monkeypatch.setattr(
        sink_probe, "tensorboard_writer", lambda: _writer_probe(store)
    )
    assert TensorBoardSink().available() is True
    assert sinks.emit(_record()) is True
    keys = {key for key, _, _ in store["scalars"]}
    assert {"schema_version", "seed"} <= keys
    assert "history.test_accuracy" in keys
    assert store["closed"] is True


def test_wandb_stub_receives_a_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the backend exists, a W&B run is opened, logged, and finished."""
    store: Dict[str, Any] = {}
    monkeypatch.setattr(sink_probe, "wandb_available", lambda: True)
    monkeypatch.setattr(
        sink_probe, "wandb_module", lambda: _wandb_probe(store)
    )
    assert WandBSink().available() is True
    assert sinks.emit(_record("wandb")) is True
    assert store["init"]["config"]["config_hash"] == "abc"
    assert store["logged"]["seed"] == 3.0
    assert store["finished"] is True


def test_requested_absent_sink_is_recorded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The local manifest is written first and names the failed sink."""
    monkeypatch.setattr(sink_probe, "tensorboard_available", lambda: False)
    monkeypatch.setattr(sink_probe, "tensorboard_writer", lambda: None)
    engine = TrainingEngine(
        dataset="mnist", num_steps=2, device="cpu", subset=2, epochs=1,
        tracking="tensorboard",
    )
    path = engine.save("sinkrun")
    assert Path(path).exists()
    stored = model_store.manifest("sinkrun")
    assert stored["tracking"]["requested"] == "tensorboard"
    assert stored["tracking"]["active"] is False
    assert "not installed" in stored["tracking"]["reason"]


def test_default_manifest_omits_optional_blocks() -> None:
    """With nothing configured, the manifest gains no new keys."""
    engine = TrainingEngine(
        dataset="mnist", num_steps=2, device="cpu", subset=2, epochs=1
    )
    engine.save("plain")
    stored = model_store.manifest("plain")
    assert "tracking" not in stored
    assert "determinism" not in stored
