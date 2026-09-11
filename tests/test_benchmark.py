"""Tests for the benchmark harness and its JSON contract."""

import json
from pathlib import Path
from typing import Any, Dict

from spikeforge.benchmark.__main__ import main
from spikeforge.benchmark.config import BenchmarkConfig, default_config
from spikeforge.benchmark.harness import run_benchmark

_FORWARD_KEYS = {
    "total_ms",
    "median_total_ms",
    "mean_ms_per_step",
    "median_ms_per_step",
    "steps_per_second",
    "repeats",
}
_MEMORY_KEYS = {
    "cuda_peak_bytes",
    "process_rss_bytes",
    "tracemalloc_peak_bytes",
}


def _config(**overrides: Any) -> BenchmarkConfig:
    """Return a tiny CPU benchmark config with any overrides applied."""
    base: Dict[str, Any] = {
        "topologies": ("fc_small", "recurrent_net"),
        "batch_size": 2,
        "steps": 4,
        "repeats": 2,
        "warmup": 1,
        "device": "cpu",
        "backward": False,
    }
    base.update(overrides)
    return BenchmarkConfig(**base)


def test_default_fixture_is_tiny() -> None:
    """The shipped default keeps CI fast."""
    config = default_config()
    assert config.batch_size <= 4
    assert config.steps <= 8
    assert config.repeats <= 3


def test_report_covers_two_topologies_and_both_modes() -> None:
    """Two topologies are measured in production and educational modes."""
    report = run_benchmark(_config())
    results = report["results"]
    assert len(results) == 4
    assert {r["topology"] for r in results} == {"fc_small", "recurrent_net"}
    assert {r["mode"] for r in results} == {"production", "educational"}
    for record in results:
        assert set(record["forward"]) >= _FORWARD_KEYS
        assert set(record["memory"]) == _MEMORY_KEYS
        assert record["compile_status"] == "eager"
        assert record["compiled"] is False
    assert isinstance(json.dumps(report), str)


def test_warmup_and_repeats_are_respected() -> None:
    """Each timed block reports the configured number of repeats."""
    report = run_benchmark(_config(repeats=3))
    for record in report["results"]:
        assert record["forward"]["repeats"] == 3


def test_backward_measured_when_requested() -> None:
    """A backward block is produced when backward timing is enabled."""
    report = run_benchmark(_config(backward=True, repeats=1))
    for record in report["results"]:
        assert record["backward"] is not None
        assert set(record["backward"]) >= _FORWARD_KEYS


def test_config_and_environment_are_reported() -> None:
    """The resolved fixture and torch environment appear in the report."""
    report = run_benchmark(_config(topologies=("fc_small",)))
    assert report["config"]["batch_size"] == 2
    assert report["config"]["topologies"][0] == "fc_small"
    assert "torch_version" in report["environment"]


def test_cli_writes_json(tmp_path: Path) -> None:
    """The module CLI returns 0 and writes a JSON report."""
    out = tmp_path / "bench.json"
    code = main(
        [
            "--topology", "fc_small",
            "--device", "cpu",
            "--steps", "3",
            "--repeats", "1",
            "--warmup", "0",
            "--no-backward",
            "--out", str(out),
        ]
    )
    assert code == 0
    assert json.loads(out.read_text(encoding="utf-8"))["results"]
