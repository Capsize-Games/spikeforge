"""Serving-mode benchmark: p50/p99 latency, throughput, cold start, memory.

PT-W6 extends :mod:`spikeforge.benchmark` past the training-step harness with
an *in-process* serving benchmark over a :class:`.DeploymentBundle`. It drives
:class:`~spikeforge.serving.session.InferenceSession` predict/stream calls,
reports percentile latency and throughput at a chosen concurrency, and emits a
record shaped like the training harness (:func:`_record`) so the existing
:class:`~spikeforge.benchmark.store.BenchmarkStore` and the regression gate in
:mod:`spikeforge.benchmark.compare` consume it unchanged.

The report is plain JSON-able data and is honest about what is unavailable:
a percentile over zero samples is ``0.0`` and an unavailable throughput is
``None`` rather than a fabricated number.
"""

from dataclasses import asdict
from typing import Any, Dict, Optional

from spikeforge.benchmark import memory
from spikeforge.benchmark.serving_config import (
    MODE,
    SAMPLE_SIZE,
    ServingBenchmarkConfig,
)
from spikeforge.benchmark.serving_report import _environment, _record
from spikeforge.benchmark.serving_run import (
    _one_call,
    _prepare,
    _step_latencies,
    _timed_run,
    percentile,
)
from spikeforge.benchmark.store import BenchmarkStore
from spikeforge.benchmark.suite import with_metadata
from spikeforge.runtime import device as device_mod

__all__ = [
    "MODE",
    "SAMPLE_SIZE",
    "ServingBenchmarkConfig",
    "percentile",
    "run_serving_benchmark",
    "run_serving_suite",
]


def run_serving_benchmark(
    config: ServingBenchmarkConfig,
) -> Dict[str, Any]:
    """Measure a bundle's in-process predict path, returning JSON-able data."""
    device = device_mod.resolve(config.device)
    bundle, session, sample, steps, cold_start = _prepare(config, device)
    durations, wall = _timed_run(config, bundle, session, sample, device)
    memory_info = memory.snapshot(lambda: _one_call(session, sample), device)
    step_durations = _step_latencies(session, sample)
    record = _record(
        config, bundle, durations, wall, steps, cold_start, memory_info,
        step_durations,
    )
    return {
        "config": asdict(config),
        "environment": _environment(device),
        "results": [record],
    }


def run_serving_suite(
    config: ServingBenchmarkConfig,
    store: Optional[BenchmarkStore] = None,
    label: Optional[str] = "serving",
    save: bool = True,
) -> Dict[str, Any]:
    """Benchmark one serving bundle and optionally persist the record."""
    report = run_serving_benchmark(config)
    enriched = with_metadata(report)
    if save:
        target = store if store is not None else BenchmarkStore()
        enriched["run_id"] = target.save(enriched, label=label)
    return enriched
