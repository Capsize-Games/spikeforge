"""Performance and memory benchmarks for the interpreter's execution modes.

This package measures wall time and memory of a topology run under
production and educational modes (and, opt-in, a compiled production path).
It is deliberately separate from ``introspection``, which describes a
recorded trajectory rather than the cost of producing one.

Use :func:`run_benchmark` from Python, or ``python -m
spikeforge.benchmark`` from a shell; both return the same
JSON-serialisable report.
"""

from spikeforge.benchmark.compare import (
    DEFAULT_THRESHOLD,
    compare_runs,
    exit_code,
)
from spikeforge.benchmark.config import BenchmarkConfig, default_config
from spikeforge.benchmark.harness import run_benchmark
from spikeforge.benchmark.serving import (
    ServingBenchmarkConfig,
    percentile,
    run_serving_benchmark,
    run_serving_suite,
)
from spikeforge.benchmark.store import BenchmarkStore, default_directory
from spikeforge.benchmark.suite import run_suite, with_metadata

__all__ = [
    "BenchmarkConfig",
    "BenchmarkStore",
    "DEFAULT_THRESHOLD",
    "ServingBenchmarkConfig",
    "compare_runs",
    "default_config",
    "default_directory",
    "exit_code",
    "percentile",
    "run_benchmark",
    "run_serving_benchmark",
    "run_serving_suite",
    "run_suite",
    "with_metadata",
]
