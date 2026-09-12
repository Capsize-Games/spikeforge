"""Issue #7 acceptance: the serving-mode benchmark and the regression gate."""

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from spikeforge.benchmark.cli import main
from spikeforge.benchmark.compare import compare_runs
from spikeforge.benchmark.serving import (
    ServingBenchmarkConfig,
    percentile,
    run_serving_benchmark,
    run_serving_suite,
)
from spikeforge.benchmark.store import BenchmarkStore
from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.encode_spec import ENCODE_SPEC_VERSION, EncodeSpec
from spikeforge.topology import registry

_TOPOLOGY = "fc_small"
_PARAMS = {"hidden": 5, "num_classes": 3}
_NUM_STEPS = 4


def _write_bundle(path: Path) -> None:
    """Write a small, deterministic ``.spkf`` bundle for the tests."""
    spec, module = registry.build_topology(_TOPOLOGY, dict(_PARAMS))
    module.eval()
    bundle = DeploymentBundle(
        manifest={
            "format": bm.BUNDLE_FORMAT,
            "version": bm.BUNDLE_VERSION,
            "spec": spec.to_dict(),
            "topology": _TOPOLOGY,
            "topology_params": dict(_PARAMS),
            "encode_spec_version": ENCODE_SPEC_VERSION,
            "num_classes": _PARAMS["num_classes"],
        },
        weights=module.state_dict(),
        encode_config=EncodeSpec(
            coding="latency", num_steps=_NUM_STEPS
        ).to_dict(),
    )
    bundle.save(str(path))


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


def _config(path: str, **overrides: Any) -> ServingBenchmarkConfig:
    """Return a small serving config for ``path``."""
    values = {
        "calls": 3,
        "concurrency": 1,
        "warmup": 1,
        "seed": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return ServingBenchmarkConfig(bundle=path, **values)


def _serving_record(
    p99_ms: float, throughput: float, rss: int = 1000
) -> Dict[str, Any]:
    """Return a minimal serving-shaped record for gate tests."""
    return {
        "topology": "fc_small",
        "mode": "serving",
        "forward": {
            "mean_ms_per_step": 1.0,
            "steps_per_second": 100.0,
        },
        "serving": {
            "p99_ms": p99_ms,
            "throughput_per_second": throughput,
        },
        "memory": {"process_rss_bytes": rss},
    }


def test_serving_benchmark_reports_latency_and_throughput(
    bundle_path: str,
) -> None:
    """A serving run reports p50/p99, throughput, cold start, and memory."""
    report = run_serving_benchmark(_config(bundle_path))
    record = report["results"][0]
    assert record["mode"] == "serving"
    assert record["topology"] == _TOPOLOGY
    assert record["forward"]["mean_ms_per_step"] is not None
    serving = record["serving"]
    assert serving["p50_ms"] <= serving["p99_ms"]
    assert serving["p99_ms"] > 0.0
    assert serving["throughput_per_second"] > 0.0
    assert serving["cold_start_ms"] > 0.0
    assert serving["peak_memory_bytes"]
    assert json.dumps(report)


def test_serving_benchmark_runs_every_call_at_concurrency(
    bundle_path: str,
) -> None:
    """Throughput mode issues exactly ``calls`` calls across the workers."""
    report = run_serving_benchmark(
        _config(bundle_path, calls=4, concurrency=2)
    )
    assert report["results"][0]["serving"]["calls"] == 4


def test_serving_suite_stores_and_reloads(
    bundle_path: str, tmp_path: Path
) -> None:
    """``run_serving_suite`` persists a record the store can reload."""
    store = BenchmarkStore(str(tmp_path / "benchmarks"))
    report = run_serving_suite(_config(bundle_path), store=store)
    run_id = report["run_id"]
    assert run_id in [item["run_id"] for item in store.list_runs()]
    loaded = store.load(run_id)
    assert loaded["results"][0]["serving"]["p99_ms"] > 0.0


def test_compare_gate_flags_a_p99_regression() -> None:
    """A doubled p99 latency breaches the configured threshold."""
    baseline = {"results": [_serving_record(10.0, 100.0)]}
    candidate = {"results": [_serving_record(20.0, 100.0)]}
    result = compare_runs(baseline, candidate, threshold=0.10)
    assert result["regressed"] is True
    assert any(case["regressed"] for case in result["cases"])


def test_compare_gate_ignores_absent_serving_metrics() -> None:
    """A training record without a serving block does not regress on p99."""
    baseline = {"results": [{"topology": "fc_small", "mode": "production"}]}
    candidate = {"results": [{"topology": "fc_small", "mode": "production"}]}
    assert compare_runs(baseline, candidate)["regressed"] is False


def test_percentile_interpolates_and_handles_edges() -> None:
    """The percentile helper interpolates and copes with tiny inputs."""
    assert percentile([1.0, 2.0, 3.0, 4.0], 50.0) == 2.5
    assert percentile([], 99.0) == 0.0
    assert percentile([5.0], 99.0) == 5.0


def test_cli_runs_a_serving_benchmark(
    bundle_path: str, tmp_path: Path
) -> None:
    """``--serving --bundle`` emits a stored-ready serving report."""
    out = tmp_path / "serving.json"
    code = main(
        [
            "--serving",
            "--bundle",
            bundle_path,
            "--calls",
            "2",
            "--warmup",
            "0",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["results"][0]["mode"] == "serving"


def test_cli_serving_requires_a_bundle() -> None:
    """Omitting ``--bundle`` in serving mode is a usage error."""
    with pytest.raises(SystemExit):
        main(["--serving"])
