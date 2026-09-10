"""Headless ``targets``, ``deploy``, ``roundtrip`` and ``ingest`` commands.

These extend the verify CLI with the Phase 5c deployment surfaces: the target
registry, a per-target deployment report, the NIR persistence round-trip, and
external graph ingestion. Every command prints JSON. ``deploy`` and
``roundtrip`` exit non-zero when their report says the result is not usable,
so they work as CI gates, and ``ingest`` surfaces the typed graph errors as a
plain message with a non-zero exit instead of a traceback.

The heavy work is delegated: :func:`~snn_interpreter.targets.report.
deployment_report` classifies a graph, :func:`~snn_interpreter.nir_bridge.
roundtrip` proves persistence fidelity, and :func:`~snn_interpreter.nir_bridge.
interpret_file` runs an imported graph. This module only shapes CLI input and
turns the results into exit statuses.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

import torch

from snn_interpreter.nir_bridge import interpret_file, roundtrip
from snn_interpreter.nir_bridge.errors import (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
    UnsupportedNodeError,
)
from snn_interpreter.simulator import input_shape
from snn_interpreter.targets.report import deployment_report
from snn_interpreter.targets.summary import target_summaries
from snn_interpreter.topology.registry import build_topology
from snn_interpreter.topology.spec import TopologySpec

#: Target used for a deployment report when the caller names none.
DEFAULT_TARGET = "reference"
#: Synthetic input shape shared by the round-trip and ingest commands.
STEPS = 8
BATCH = 2
FEATURES = 784
SEED = 0

#: Graph errors ``ingest`` reports cleanly rather than letting them escape.
GraphError = (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
    UnsupportedNodeError,
)


def _spikes(spec: TopologySpec) -> torch.Tensor:
    """Return a deterministic flat spike volume shaped for ``spec``."""
    torch.manual_seed(SEED)
    flat = torch.rand(STEPS, BATCH, FEATURES)
    return input_shape.to_input_shape(flat, spec)


def targets_payload() -> Dict[str, Any]:
    """Return the availability-annotated target registry."""
    return {"targets": target_summaries()}


def deploy_report(
    topology: str,
    target: str = DEFAULT_TARGET,
    dataset: Optional[str] = None,
    sample: int = 0,
) -> Dict[str, Any]:
    """Return the deployment report for ``topology`` against ``target``.

    A ``dataset`` attaches the drift-validation section by encoding one real
    sample. Without it the report classifies the graph structure alone, which
    keeps the command offline and fast.
    """
    if dataset is None:
        spec, _ = build_topology(topology)
        return deployment_report(spec, target)
    from snn_interpreter.cli import verify

    spec, module, spikes = verify.sample_input(topology, dataset, sample)
    return deployment_report(spec, target, module=module, spikes=spikes)


def deploy_exit(report: Dict[str, Any]) -> int:
    """Return the process status for a deployment report."""
    return 0 if report["deployable"] else 1


def roundtrip_report(
    topology: str,
    out: Optional[str] = None,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Persist a topology's graph, reload it, and report fidelity.

    ``graph`` overrides the persisted artifact so a perturbed parameter shows
    as drift; ``out`` chooses the file, else a temporary one is used.
    """
    spec, module = build_topology(topology)
    return roundtrip(spec, module, _spikes(spec), graph=graph, path=out)


def roundtrip_exit(report: Dict[str, Any]) -> int:
    """Return the process status for a round-trip fidelity report."""
    return 0 if report["identical"] else 1


def _readout(result: Any) -> List[float]:
    """Return the interpreter's readout as a plain float list."""
    return [float(value) for value in result.readout.reshape(-1)]


def ingest_summary(path: str, topology: str) -> Dict[str, Any]:
    """Load an external graph, run it on a shaped input, and summarize it.

    Errors the loader raises are typed and left to the caller; the summary
    names the traced nodes so an ingest is never silently partial.
    """
    spec, _ = build_topology(topology)
    result = interpret_file(path, _spikes(spec))
    return {
        "path": path,
        "topology": topology,
        "steps": int(result.steps),
        "readout": _readout(result),
        "spike_nodes": sorted(result.spikes),
        "membrane_nodes": sorted(result.membranes),
    }


def _run_targets(args: argparse.Namespace) -> int:
    """Print the target registry as JSON."""
    print(json.dumps(targets_payload(), indent=2))
    return 0


def _run_deploy(args: argparse.Namespace) -> int:
    """Print a deployment report and return its usability status."""
    report = deploy_report(
        args.topology, args.target, args.dataset, args.sample
    )
    print(json.dumps(report, indent=2))
    return deploy_exit(report)


def _run_roundtrip(args: argparse.Namespace) -> int:
    """Print the NIR round-trip report and return its fidelity status."""
    report = roundtrip_report(args.topology, args.out)
    print(json.dumps(report, indent=2))
    return roundtrip_exit(report)


def _run_ingest(args: argparse.Namespace) -> int:
    """Print an ingest summary, or the typed graph error, plus a status."""
    try:
        summary = ingest_summary(args.file, args.topology)
    except GraphError as error:
        print(str(error))
        return 1
    print(json.dumps(summary, indent=2))
    return 0


def add_subcommands(subs: Any) -> None:
    """Register the Phase 5c deployment subcommands on ``subs``."""
    listing = subs.add_parser("targets", help="list deployment targets")
    listing.set_defaults(handler=_run_targets)

    deploy = subs.add_parser("deploy", help="report a topology's target fit")
    deploy.add_argument("--topology", default="conv_net")
    deploy.add_argument("--target", default=DEFAULT_TARGET)
    deploy.add_argument("--dataset", default=None)
    deploy.add_argument("--sample", type=int, default=0)
    deploy.set_defaults(handler=_run_deploy)

    trip = subs.add_parser("roundtrip", help="round-trip a graph via disk")
    trip.add_argument("--topology", default="conv_net")
    trip.add_argument("--out", default=None)
    trip.set_defaults(handler=_run_roundtrip)

    ingest = subs.add_parser("ingest", help="run an external NIR graph")
    ingest.add_argument("--file", required=True)
    ingest.add_argument("--topology", default="conv_net")
    ingest.set_defaults(handler=_run_ingest)
