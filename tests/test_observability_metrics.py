"""Metrics registry recording, JSON snapshot, and training wiring."""

import json
from typing import Iterator

import pytest

from snn_interpreter.observability import metrics
from snn_interpreter.observability.registry import MetricsRegistry
from snn_interpreter.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _clean_metrics() -> Iterator[None]:
    """Reset the shared registry before and after every test."""
    metrics.reset()
    yield
    metrics.reset()


def test_counter_gauge_and_timer_snapshot() -> None:
    """Counters accumulate, gauges overwrite, and timers summarise."""
    registry = MetricsRegistry()
    registry.counter("steps")
    registry.counter("steps", 2)
    registry.gauge("accuracy", 91.5)
    with registry.timer("encode"):
        pass
    snapshot = registry.snapshot()
    assert snapshot["counters"]["steps"] == 3.0
    assert snapshot["gauges"]["accuracy"] == 91.5
    timer = snapshot["timers"]["encode"]
    assert timer["count"] == 1
    assert timer["total_seconds"] >= 0.0
    assert json.dumps(snapshot)


def test_snapshot_is_json_dumpable_and_resettable() -> None:
    """The snapshot serialises to JSON and clears on reset."""
    metrics.counter("runs")
    metrics.gauge("loss", 0.25)
    metrics.observe("forward", 0.01)
    assert json.dumps(metrics.snapshot())
    metrics.reset()
    assert metrics.snapshot() == {
        "counters": {}, "gauges": {}, "timers": {}
    }


def test_training_and_validation_increment_metrics() -> None:
    """A short training run records the expected counters and timers."""
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        subset=3,
        batch_size=4,
        epochs=1,
    )
    for _ in engine.train():
        pass
    snapshot = metrics.snapshot()
    assert snapshot["counters"]["train.steps"] >= 1
    assert snapshot["counters"]["validation.runs"] >= 1
    assert "train.encode_seconds" in snapshot["timers"]
    assert "train.forward_seconds" in snapshot["timers"]
    assert "train.backward_seconds" in snapshot["timers"]
    assert "validation.seconds" in snapshot["timers"]
    assert "validation.accuracy" in snapshot["gauges"]
