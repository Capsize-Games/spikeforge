"""Command-line interface for the benchmark suite.

``python -m snn_interpreter.benchmark`` (or the ``snn-benchmark`` script)
runs a fixture and prints JSON. ``--save``/``--list`` persist and list runs
through the :class:`~snn_interpreter.benchmark.store.BenchmarkStore`, and
``--compare`` diffs a stored baseline against a fresh or stored run. When
``--fail-on-regression`` is given the command exits non-zero on a regression,
so it works as a CI gate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from snn_interpreter.benchmark.compare import (
    DEFAULT_THRESHOLD,
    compare_runs,
    exit_code,
)
from snn_interpreter.benchmark.config import BenchmarkConfig
from snn_interpreter.benchmark.harness import run_benchmark
from snn_interpreter.benchmark.store import BenchmarkStore
from snn_interpreter.benchmark.suite import with_metadata


def _add_fixture_args(parser: argparse.ArgumentParser) -> None:
    """Register the fixture-shaping arguments."""
    parser.add_argument("--topology", action="append", dest="topologies")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compiled", action="store_true")
    parser.add_argument("--no-backward", action="store_true")
    parser.add_argument("--out", default=None)


def _add_store_args(parser: argparse.ArgumentParser) -> None:
    """Register the store, listing, and comparison arguments."""
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--label", default=None)
    parser.add_argument("--store-dir", dest="store_dir", default=None)
    parser.add_argument("--list", action="store_true", dest="list_runs")
    parser.add_argument("--compare", default=None)
    parser.add_argument("--against", default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--fail-on-regression", action="store_true")


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="snn-benchmark",
        description="Benchmark interpreter execution modes.",
    )
    _add_fixture_args(parser)
    _add_store_args(parser)
    return parser


def _config(args: argparse.Namespace) -> BenchmarkConfig:
    """Build a :class:`BenchmarkConfig` from parsed CLI arguments."""
    return BenchmarkConfig(
        topologies=tuple(args.topologies or ("fc_small",)),
        batch_size=args.batch_size,
        steps=args.steps,
        repeats=args.repeats,
        warmup=args.warmup,
        seed=args.seed,
        device=args.device,
        compiled=args.compiled,
        backward=not args.no_backward,
    )


def _store(args: argparse.Namespace) -> BenchmarkStore:
    """Return the store configured by ``--store-dir``."""
    return BenchmarkStore(args.store_dir)


def _emit(payload: Mapping[str, Any], out: Optional[str]) -> int:
    """Print ``payload`` or write it to ``out``; return success."""
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _run_list(args: argparse.Namespace) -> int:
    """Print the stored run summaries as JSON."""
    return _emit({"runs": _store(args).list_runs()}, args.out)


def _run_fixture(args: argparse.Namespace) -> int:
    """Run the fixture, optionally save it, and emit the report."""
    report = run_benchmark(_config(args))
    if args.save:
        report = with_metadata(report)
        report["run_id"] = _store(args).save(report, label=args.label)
    return _emit(report, args.out)


def _candidate(args: argparse.Namespace) -> Dict[str, Any]:
    """Return a stored run when ``--against`` is set, else a fresh run."""
    if args.against:
        return _store(args).load(args.against)
    return run_benchmark(_config(args))


def _run_compare(args: argparse.Namespace) -> int:
    """Compare a baseline run and print the deltas plus a status."""
    baseline = _store(args).load(args.compare)
    result = compare_runs(baseline, _candidate(args), args.threshold)
    print(json.dumps(result, indent=2))
    return exit_code(result) if args.fail_on_regression else 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch list, compare, or a fixture run."""
    args = _parser().parse_args(argv)
    if args.list_runs:
        return _run_list(args)
    if args.compare:
        return _run_compare(args)
    return _run_fixture(args)


if __name__ == "__main__":
    sys.exit(main())
