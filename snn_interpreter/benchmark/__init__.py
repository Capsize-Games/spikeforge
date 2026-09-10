"""Performance and memory benchmarks for the interpreter's execution modes.

This package measures wall time and memory of a topology run under
production and educational modes (and, opt-in, a compiled production path).
It is deliberately separate from ``introspection``, which describes a
recorded trajectory rather than the cost of producing one.

Use :func:`run_benchmark` from Python, or ``python -m
snn_interpreter.benchmark`` from a shell; both return the same
JSON-serialisable report.
"""

from snn_interpreter.benchmark.config import BenchmarkConfig, default_config
from snn_interpreter.benchmark.harness import run_benchmark

__all__ = ["BenchmarkConfig", "default_config", "run_benchmark"]
