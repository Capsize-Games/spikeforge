"""Headless ``rewrite`` and ``run`` backend commands.

These extend the verify/targets CLIs with the executed view of a deployment:
``rewrite`` applies a target's declared substitutions to the configured
topology's NIR graph and prints the rewrite report (with a post-rewrite drift
check), while ``run`` compiles the target-ready graph, runs it on the backend,
and prints the backend result. ``run`` exits non-zero unless the status is
``ok`` and the comparison to the reference interpreter is within tolerance,
so it works as a CI gate.

Both commands build a deterministic synthetic spike fixture, so they stay
offline and reproducible.
"""

import argparse
import json
from typing import Any, Dict, Optional, Tuple

import torch

from snn_interpreter.cli import fixture
from snn_interpreter.nir_bridge.exporter import to_nir
from snn_interpreter.targets.backends import STATUS_OK, compile_run
from snn_interpreter.targets.backends.result import BackendResult
from snn_interpreter.targets.rewrite import rewrite
from snn_interpreter.targets.rewrite_drift import (
    READOUT_MAX_ABS,
    SPIKE_AGREEMENT,
    SPIKE_MAX_ABS,
)
from snn_interpreter.targets.rewrite_result import RewriteResult

#: Target used when the caller names none.
DEFAULT_TARGET = "reference"
#: Synthetic input shape shared by both commands.
STEPS = 8
BATCH = 2
SEED = 0

_Fixture = Tuple[Any, torch.Tensor]


def _graph(topology: str) -> _Fixture:
    """Return ``(graph, shaped_spikes)`` for a deterministic fixture.

    The fixture is shaped by the shared :mod:`snn_interpreter.cli.fixture`
    helper, so a sequence topology receives a ``[T, B, L, D]`` frame instead
    of a flat image volume.
    """
    spec, module, spikes = fixture.synthetic_input(
        topology, STEPS, BATCH, SEED
    )
    return to_nir(spec, module), spikes


def rewrite_report(topology: str, target: str) -> RewriteResult:
    """Return the rewrite result for ``topology`` against ``target``."""
    graph, spikes = _graph(topology)
    return rewrite(graph, target, spikes)


def run_result(topology: str, target: str) -> BackendResult:
    """Return the backend run result for ``topology`` against ``target``."""
    graph, spikes = _graph(topology)
    return compile_run(target, graph, spikes)


def _within(compare: Optional[Dict[str, Any]]) -> bool:
    """Return True when a backend comparison stays within tolerance."""
    if not compare:
        return False
    readout = compare.get("readout") or {}
    spikes = compare.get("spikes") or {}
    return (
        readout.get("max_abs", 1.0) <= READOUT_MAX_ABS
        and spikes.get("max_abs", 1.0) <= SPIKE_MAX_ABS
        and spikes.get("agreement", 0.0) >= SPIKE_AGREEMENT
    )


def rewrite_exit(report: RewriteResult) -> int:
    """Return the process status for a rewrite report (always success)."""
    return 0


def run_exit(result: BackendResult) -> int:
    """Return the process status for a backend run result."""
    if result.status != STATUS_OK:
        return 1
    return 0 if _within(dict(result.compare or {})) else 1


def _run_rewrite(args: argparse.Namespace) -> int:
    """Print the rewrite report JSON."""
    result = rewrite_report(args.topology, args.target)
    print(json.dumps(result.report.to_dict(), indent=2))
    return rewrite_exit(result)


def _run_run(args: argparse.Namespace) -> int:
    """Print the backend result JSON and return its usability status."""
    result = run_result(args.topology, args.target)
    print(json.dumps(result.to_dict(), indent=2))
    return run_exit(result)


def add_subcommands(subs: Any) -> None:
    """Register the ``rewrite`` and ``run`` backend subcommands."""
    rw = subs.add_parser("rewrite", help="apply a target's substitutions")
    rw.add_argument("--topology", default="conv_net")
    rw.add_argument("--target", default=DEFAULT_TARGET)
    rw.set_defaults(handler=_run_rewrite)

    run = subs.add_parser("run", help="compile and run a backend")
    run.add_argument("--topology", default="conv_net")
    run.add_argument("--target", default=DEFAULT_TARGET)
    run.set_defaults(handler=_run_run)
