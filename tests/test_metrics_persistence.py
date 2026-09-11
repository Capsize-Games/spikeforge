"""Persisted metrics: round-trip, reload, and the default-off path."""

from typing import Any, Iterator

import pytest

from spikeforge.observability import metrics, persistence
from spikeforge.observability.persistence import MetricsPersistence
from spikeforge.observability.snapshot import MetricSnapshot
from spikeforge.observability.store import SnapshotStore, safe_run_id


@pytest.fixture(autouse=True)
def _clean(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Isolate the registry and persistence from the real data directory."""
    metrics.reset()
    monkeypatch.delenv(persistence.ENV_FLAG, raising=False)
    persistence.reset_default()
    yield
    metrics.reset()
    persistence.reset_default()


def test_enabled_flush_survives_a_simulated_restart(tmp_path: Any) -> None:
    """A flushed snapshot reloads from disk after the process restarts."""
    store = MetricsPersistence(
        root=str(tmp_path), enabled=True, run_id="run1"
    )
    metrics.counter("steps", 3)
    metrics.gauge("accuracy", 42.0)
    written = store.flush()
    assert written is not None
    restarted = MetricsPersistence(
        root=str(tmp_path), enabled=True, run_id="run1"
    )
    loaded = restarted.load()
    assert loaded is not None
    assert loaded.run_id == "run1"
    assert loaded.metrics["counters"]["steps"] == 3.0
    assert loaded.metrics["gauges"]["accuracy"] == 42.0
    assert loaded.timestamp == written.timestamp
    assert restarted.status()["last_flush"] is None


def test_disabled_by_default_writes_nothing(tmp_path: Any) -> None:
    """Without the opt-in flag, flush is a no-op and no file appears."""
    store = MetricsPersistence(root=str(tmp_path), run_id="run1")
    assert store.enabled is False
    metrics.counter("steps")
    assert store.flush() is None
    assert store.load() is None
    assert list(tmp_path.iterdir()) == []


def test_env_flag_and_dir_override(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPIKEFORGE_METRICS_PERSIST enables writing under the env dir."""
    monkeypatch.setenv(persistence.ENV_FLAG, "1")
    monkeypatch.setenv("SPIKEFORGE_METRICS_DIR", str(tmp_path / "metrics"))
    persistence.reset_default()
    metrics.counter("env.steps", 2)
    assert persistence.flush(run_id="envrun") is not None
    loaded = persistence.load(run_id="envrun")
    assert loaded is not None
    assert loaded.metrics["counters"]["env.steps"] == 2.0
    assert persistence.status()["enabled"] is True


def test_store_round_trips_and_lists_runs(tmp_path: Any) -> None:
    """The store writes, reads, and enumerates snapshot files."""
    store = SnapshotStore(root=str(tmp_path))
    store.write(MetricSnapshot.capture("a", {"counters": {"x": 1.0}}, 1.0))
    assert store.list_runs() == ["a"]
    loaded = store.read("a")
    assert loaded is not None
    assert loaded.metrics["counters"]["x"] == 1.0
    assert loaded.timestamp == 1.0
    assert store.read("missing") is None


def test_safe_run_id_sanitises_names() -> None:
    """A run id is made safe for a file name."""
    assert safe_run_id("bad id/one") == "bad_id_one"
    assert safe_run_id("   ") == "default"
