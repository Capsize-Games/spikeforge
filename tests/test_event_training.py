"""Event-dataset training: end to end, honest, and fully offline.

The synthetic source is used for the offline path and is explicitly labelled
as ``synthetic`` in the checkpoint. The tonic branch is exercised without the
network by stubbing the loader, so both provenance paths are covered here.
"""

from typing import Any, Dict, Tuple

import pytest
import torch

from server import training as training_service
from spikeforge.data import event_loader
from spikeforge.data.datasets import build_dataset
from spikeforge.data.event_errors import EventsExtraMissingError
from spikeforge.data.event_geometry import EventGeometryError
from spikeforge.events import event_source, synthetic
from spikeforge.events.event_sample import EventSample
from spikeforge.events.event_source import EventSampleSource
from spikeforge.network import model_store
from spikeforge.topology import registry
from spikeforge.training.event_batches import event_batches
from spikeforge.training.event_engine import EventTrainingEngine
from spikeforge.training.training_engine import TrainingEngine

_STEPS = 4


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _engine(**kwargs: Any) -> EventTrainingEngine:
    """Return an offline CPU engine with small, download-free defaults."""
    defaults: Dict[str, Any] = {
        "dataset": "n_mnist", "synthetic_only": True, "num_steps": _STEPS,
        "subset": 10, "batch_size": 4, "hidden": 8, "epochs": 1,
        "device": "cpu",
    }
    defaults.update(kwargs)
    return EventTrainingEngine(**defaults)


def _fake_pair(
    name: str, index: int, num_steps: int, save_to: Any = None,
    split: str = "train",
) -> Tuple[EventSample, int]:
    """Return a deterministic synthetic sample in the loader's shape."""
    sample = synthetic.moving_dot(
        num_steps=num_steps, shape=(28, 28), seed=index
    )
    return sample, index % 4


def test_event_batches_bridge_the_simulator_contract() -> None:
    """Bridged events are time-major [T, B, F] spikes plus [B] labels."""
    source = EventSampleSource(
        "n_mnist", synthetic_only=True, num_steps=_STEPS
    )
    spec, _module = registry.build_topology(
        "fc_legacy", {"hidden": 8, "input_size": 784, "num_classes": 4}
    )
    spikes, labels = event_batches(source, spec, subset=10, batch_size=3)[0]
    assert list(spikes.shape) == [_STEPS, 3, 784]
    assert list(labels.shape) == [3]


def test_event_engine_trains_one_epoch() -> None:
    """One epoch of bridged events streams finite training metrics."""
    torch.manual_seed(0)
    points = list(_engine().train())
    assert len(points) == 1
    assert points[0]["loss"] > 0.0
    assert points[0]["epoch"] == 0
    assert 0.0 <= points[0]["train_accuracy"] <= 1.0


def test_event_engine_trains_a_conv_topology() -> None:
    """Bridged [T,B,1,28,28] frames drive the conv_net simulator loop."""
    torch.manual_seed(0)
    engine = _engine(topology="conv_net", topology_params={"channels": 2})
    inputs, targets = engine._epoch_batches()[0]
    assert list(inputs.shape) == [_STEPS, 4, 1, 28, 28]
    metrics = engine._train_batch(inputs, targets)
    assert metrics["loss"] > 0.0


def test_event_checkpoint_records_modality() -> None:
    """A saved event model names its modality and its synthetic origin."""
    torch.manual_seed(0)
    _engine().save("event_fixture")
    meta = model_store.load("event_fixture")["meta"]
    assert meta["modality"] == "event"
    assert meta["event_origin"] == "synthetic"
    assert "synthetic" in meta["event_description"]


def test_event_checkpoint_round_trips_weights() -> None:
    """An event-trained checkpoint reloads with identical weights."""
    torch.manual_seed(0)
    engine = _engine()
    list(engine.train())
    engine.save("event_round_trip")
    restored = _engine(checkpoint="event_round_trip")
    for key, value in engine.net.state_dict().items():
        assert torch.equal(value, restored.net.state_dict()[key])
    inputs, targets = restored._epoch_batches()[0]
    assert restored._train_batch(inputs, targets)["loss"] > 0.0


def test_missing_events_extra_is_named(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without tonic, training raises the typed error naming the extra."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: False)
    with pytest.raises(EventsExtraMissingError) as ctx:
        EventTrainingEngine(dataset="n_mnist", num_steps=_STEPS, device="cpu")
    message = str(ctx.value)
    assert "tonic" in message
    assert "events" in message
    assert "n_mnist" in message


def test_tonic_backed_path_uses_the_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With tonic present the engine loads through the event loader."""
    monkeypatch.setattr(event_source, "dataset_available", lambda name: True)
    monkeypatch.setattr(event_loader, "load_event_pair", _fake_pair)
    engine = _engine(synthetic_only=False)
    assert engine._event_source.origin == event_source.TONIC
    points = list(engine.train())
    assert points and points[0]["loss"] > 0.0


def test_geometry_mismatch_is_reported_for_conv() -> None:
    """A conv topology rejects a sensor of the wrong side, by name."""
    with pytest.raises(EventGeometryError) as ctx:
        _engine(topology="conv_net", topology_params={"input_size": 32})
    message = str(ctx.value)
    assert "conv_net" in message
    assert "32x32" in message
    assert "28x28" in message


def test_geometry_mismatch_is_reported_for_features() -> None:
    """A feature topology rejects a sensor whose area misses input_size."""
    with pytest.raises(EventGeometryError) as ctx:
        _engine(topology="fc_legacy", topology_params={"input_size": 100})
    message = str(ctx.value)
    assert "100" in message
    assert "784" in message


def test_event_engine_evaluates_and_predicts() -> None:
    """Held-out scoring and the sample prediction use bridged events."""
    torch.manual_seed(0)
    engine = _engine()
    assert 0.0 <= engine.evaluate() <= 100.0
    sample = engine.predict_sample()
    assert len(sample["digits"]) == 4
    assert len(sample["labels"]) == 4


def test_training_factory_routes_by_modality() -> None:
    """Event datasets select the event engine; images keep the old one."""
    assert training_service._engine_class("n_mnist") is EventTrainingEngine
    assert training_service._engine_class("mnist") is TrainingEngine


def test_build_dataset_names_the_event_path() -> None:
    """The image builder refuses an event dataset with a named reason."""
    with pytest.raises(ValueError) as ctx:
        build_dataset("n_mnist")
    assert "event" in str(ctx.value)
