"""Benchmark store round-trips and the suite persistence contract."""

import json
from pathlib import Path

import pytest

from snn_interpreter.benchmark.config import BenchmarkConfig
from snn_interpreter.benchmark.store import BenchmarkStore, default_directory
from snn_interpreter.benchmark.suite import run_suite


def _config() -> BenchmarkConfig:
    """Return a tiny CPU fixture with no backward pass."""
    return BenchmarkConfig(
        topologies=("fc_small",),
        batch_size=2,
        steps=3,
        repeats=1,
        warmup=0,
        device="cpu",
        backward=False,
    )


def test_store_save_load_list_round_trip(tmp_path: Path) -> None:
    """A saved record is writable, loadable, and summarised when listed."""
    store = BenchmarkStore(str(tmp_path))
    run_id = store.save(
        {"config": {"x": 1}, "results": [1, 2]}, label="alpha"
    )
    assert store.path_for(run_id).exists()
    loaded = store.load(run_id)
    assert loaded["run_id"] == run_id
    assert loaded["label"] == "alpha"
    runs = store.list_runs()
    assert runs[0]["run_id"] == run_id
    assert runs[0]["result_count"] == 2


def test_store_missing_run_raises(tmp_path: Path) -> None:
    """Loading an unknown run raises ``FileNotFoundError``."""
    store = BenchmarkStore(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        store.load("does_not_exist")


def test_default_directory_is_under_data_dir(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without an override the store lives under the data directory."""
    monkeypatch.delenv("SNN_BENCHMARK_DIR", raising=False)
    assert default_directory().endswith("benchmarks")


def test_run_suite_persists_multiple_configs(tmp_path: Path) -> None:
    """The suite records one result per topology and persists the report."""
    store = BenchmarkStore(str(tmp_path))
    report = run_suite(
        topologies=("fc_small", "recurrent_net"),
        modes=("production",),
        config=_config(),
        store=store,
        label="suite",
    )
    assert report["run_id"]
    assert report["versions"]["python"]
    assert report["created_at"] > 0
    assert {r["topology"] for r in report["results"]} == {
        "fc_small",
        "recurrent_net",
    }
    assert all(r["mode"] == "production" for r in report["results"])
    stored = store.load(report["run_id"])
    assert len(stored["results"]) == 2
    text = store.path_for(report["run_id"]).read_text(encoding="utf-8")
    assert json.loads(text)["run_id"] == report["run_id"]


def test_run_suite_can_skip_saving(tmp_path: Path) -> None:
    """``save=False`` returns the report without touching the store."""
    store = BenchmarkStore(str(tmp_path))
    report = run_suite(
        topologies=("fc_small",),
        modes=("educational",),
        config=_config(),
        store=store,
        save=False,
    )
    assert "run_id" not in report
    assert store.list_runs() == []
