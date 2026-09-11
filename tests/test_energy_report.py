"""The JSON-able energy report and its topology convenience wrapper."""

import json
from typing import Any, Dict

from spikeforge.benchmark.config import BenchmarkConfig
from spikeforge.benchmark.harness import run_benchmark
from spikeforge_targets.energy import account, measure_topology
from spikeforge_targets.energy.report import EnergyReport
from spikeforge_targets.event_runtime.sparse_result import SparseResult

_COUNTS = {"sop": 20, "mac": 80, "ac": 10, "timesteps": 4}


def _benchmark_config(**overrides: Any) -> BenchmarkConfig:
    """Return a tiny CPU benchmark fixture with any overrides applied."""
    base: Dict[str, Any] = {
        "topologies": ("fc_small",), "steps": 2, "batch_size": 1,
        "repeats": 1, "warmup": 0, "device": "cpu", "backward": False,
    }
    base.update(overrides)
    return BenchmarkConfig(**base)


def test_report_dict_has_the_documented_shape() -> None:
    """The report dict carries the plan's keys, all JSON-able."""
    report = account(_COUNTS, "lava_loihi2")
    payload = report.to_dict()
    assert set(payload) >= {
        "target", "estimate", "basis", "timesteps", "ops",
        "efficiency", "energy", "latency", "measured", "notes",
    }
    assert payload["target"] == "lava_loihi2"
    assert payload["ops"] == {"sop": 20, "mac": 80, "ac": 10}
    assert json.loads(json.dumps(payload)) == payload


def test_unavailable_report_is_jsonable_with_nulls() -> None:
    """An unavailable target serializes with null energy and latency."""
    payload = account(_COUNTS, "no_such_target").to_dict()
    assert payload["energy"] is None
    assert payload["latency"] is None
    assert json.dumps(payload)


def test_report_accepts_a_sparse_result_directly() -> None:
    """A SparseResult is accepted in place of a raw counts mapping."""
    result, report = measure_topology(
        "conv_net", "reference", steps=2, batch=1
    )
    assert isinstance(result, SparseResult)
    assert isinstance(report, EnergyReport)
    assert report.ops["sop"] == result.sop
    assert report.ops["mac"] == result.mac
    assert report.timesteps == result.steps


def test_benchmark_energy_block_is_opt_in() -> None:
    """The harness attaches an energy block only when asked, additively."""
    off = run_benchmark(_benchmark_config())
    assert off["results"][0]["energy"] is None
    assert off["config"]["energy"] is False

    on = run_benchmark(_benchmark_config(energy=True, energy_target="norse"))
    record = on["results"][0]
    assert {
        "topology", "mode", "compiled", "compile_status",
        "forward", "backward", "memory",
    } <= set(record)
    block = record["energy"]
    assert block["report"]["target"] == "norse"
    assert block["report"]["basis"] == "declared cost table"
    assert block["counts"]["sop"] < block["counts"]["mac"]
    assert json.dumps(record)
