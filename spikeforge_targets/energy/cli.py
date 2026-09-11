"""The ``spikeforge-energy`` console script: ``account`` and ``report``.

``account`` prints the energy/latency report for one topology and target;
``report`` additionally runs the dense path once and attaches the sparse-vs-
dense parity block, writing to ``--out`` when given. Without ``--sparse`` the
report is the dense baseline (``sop == mac``); with it the event-driven
reduction is reported. Every number is an estimate unless a device reports.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from spikeforge.simulator.runner import run
from spikeforge_targets.energy.accounting import (
    BATCH,
    SEED,
    STEPS,
    account_spikes,
    measure_topology,
    topology_fixture,
)
from spikeforge_targets.event_runtime.dense_compare import compare


def _emit(payload: Dict[str, Any], out: Optional[str]) -> int:
    """Print ``payload`` or write it to ``out``; return success."""
    text = json.dumps(payload, indent=2)
    if out:
        Path(out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _run_account(args: argparse.Namespace) -> int:
    """Print the energy/latency report for the requested target."""
    _, report = measure_topology(
        args.topology, args.target, args.steps, args.batch_size,
        args.seed, args.sparse,
    )
    return _emit(report.to_dict(), args.out)


def _run_report(args: argparse.Namespace) -> int:
    """Print the report plus the sparse-vs-dense parity comparison."""
    _spec, module, spikes = topology_fixture(
        args.topology, args.steps, args.batch_size, args.seed
    )
    result, report = account_spikes(module, spikes, args.target, args.sparse)
    with torch.no_grad():
        dense = run(module, spikes)
    payload = {
        "report": report.to_dict(),
        "comparison": compare(result, dense),
    }
    return _emit(payload, args.out)


def _add_fixture_args(parser: argparse.ArgumentParser) -> None:
    """Register the fixture and target arguments shared by both commands."""
    parser.add_argument("--topology", default="conv_net")
    parser.add_argument("--target", default="reference")
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sparse", action="store_true")
    parser.add_argument("--out", default=None)


def add_subcommands(subs: Any) -> None:
    """Register the ``account`` and ``report`` subcommands on ``subs``."""
    account_cmd = subs.add_parser(
        "account", help="account ops and energy for a target"
    )
    _add_fixture_args(account_cmd)
    account_cmd.set_defaults(handler=_run_account)

    report_cmd = subs.add_parser(
        "report", help="account and compare the dense baseline"
    )
    _add_fixture_args(report_cmd)
    report_cmd.set_defaults(handler=_run_report)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the energy CLI."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-energy",
        description="Account event-driven operations as energy and latency.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
