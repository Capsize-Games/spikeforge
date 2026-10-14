#!/usr/bin/env python3
"""Compute the ARCH-0001 topology triggers (T1-T4) from local evidence.

See ``plans/arch-0001-decision-metrics.md`` for the trigger definitions. The
script is deliberately stdlib-first and read-only: it derives what it can from
``git log`` and the pushed ``*-v*`` tags, and augments the picture with GitHub
Actions timing through the ``gh`` CLI when that is available.

Every input degrades to ``None`` (reported as "unavailable") rather than
raising, so the script runs offline, in a shallow clone, and in CI. A trigger
with an unavailable input never fires; the printed status names the missing
input so the maintainer can record the decision in the ADR.

Usage::

    python scripts/topology_metrics.py [--window-days 90] [--as-of DATE]

It prints the metric values, then, for each named trigger, whether it fired and
which clauses held. ``--json`` emits the same data as JSON.
"""

import argparse
from datetime import datetime, timezone
from typing import Optional, Sequence

from topology_metrics_git import _parse_stamp
from topology_metrics_metrics import _collect
from topology_metrics_output import _print_json, _print_report
from topology_metrics_triggers import _build_triggers

#: Trailing rolling window used for the "releases" clause of every trigger.
DEFAULT_WINDOW_DAYS = 90

def _parse_as_of(value: Optional[str]) -> datetime:
    """Return the window anchor, defaulting to the current UTC time."""
    if not value:
        return datetime.now(timezone.utc)
    parsed = _parse_stamp(value)
    if parsed is None:
        raise SystemExit(f"invalid --as-of value: {value!r}")
    return parsed


def _add_window_args(parser: argparse.ArgumentParser) -> None:
    """Add the rolling-window selection flags."""
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help=f"rolling window in days (default: {DEFAULT_WINDOW_DAYS})",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        help="window anchor as an ISO date (default: now, UTC)",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Compute the ARCH-0001 topology triggers T1-T4."
    )
    _add_window_args(parser)
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the metrics and statuses as JSON",
    )
    return parser


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the command-line arguments."""
    return _build_parser().parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Compute and print the metrics and trigger statuses."""
    args = _parse_args(argv)
    as_of = _parse_as_of(args.as_of)
    metrics = _collect(as_of, args.window_days)
    triggers = _build_triggers(metrics)
    if args.json:
        _print_json(metrics, triggers, args.window_days, as_of)
    else:
        _print_report(metrics, triggers, args.window_days, as_of)
    return 0


if __name__ == "__main__":
    main()
