"""CLI entry point: ``python -m snn_interpreter.benchmark``."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from snn_interpreter.benchmark.config import BenchmarkConfig
from snn_interpreter.benchmark.harness import run_benchmark


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the benchmark CLI."""
    parser = argparse.ArgumentParser(
        prog="snn-benchmark",
        description="Benchmark interpreter execution modes.",
    )
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


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv``, run the benchmark, and print or write JSON."""
    args = _parser().parse_args(argv)
    text = json.dumps(run_benchmark(_config(args)), indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
