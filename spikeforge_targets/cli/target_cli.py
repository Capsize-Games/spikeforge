"""Headless ``targets``, ``deploy``, ``roundtrip`` and ``ingest`` commands.

These extend the verify CLI with the Phase 5c deployment surfaces: the target
registry, a per-target deployment report, the NIR persistence round-trip, and
external graph ingestion. Every command prints JSON. ``deploy`` and
``roundtrip`` exit non-zero when their report says the result is not usable,
so they work as CI gates, and ``ingest`` surfaces the typed graph errors as a
plain message with a non-zero exit instead of a traceback.

The heavy work is delegated: :func:`~spikeforge_targets.report.
deployment_report` classifies a graph, :func:`~spikeforge.nir_bridge.
roundtrip` proves persistence fidelity, and :func:`~spikeforge.nir_bridge.
interpret_file` runs an imported graph. This module only shapes CLI input and
turns the results into exit statuses.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from spikeforge.cli import backend_cli, extract_cli, fixture
from spikeforge.nir_bridge import interpret_file, roundtrip
from spikeforge.nir_bridge.errors import (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
    UnsupportedNodeError,
)
from spikeforge_targets.report import deployment_report
from spikeforge_targets.summary import target_summaries

#: Target used for a deployment report when the caller names none.
DEFAULT_TARGET = "reference"
#: Synthetic input shape shared by the round-trip and ingest commands.
STEPS = 8
BATCH = 2
SEED = 0

#: Graph errors ``ingest`` reports cleanly rather than letting them escape.
GraphError = (
    GraphNotFoundError,
    MalformedGraphError,
    UnknownNodeKindError,
    UnsupportedNodeError,
)


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

    A ``dataset`` selects the sample whose spikes drive the validation and
    substitution sections; without it a deterministic synthetic fixture is
    used, so the report still carries the executed substitution view while
    staying offline.
    """
    if dataset is None:
        spec, module, spikes = fixture.synthetic_input(
            topology, STEPS, BATCH, SEED
        )
        return deployment_report(
            spec, target, module=module, spikes=spikes
        )
    from spikeforge.cli import verify

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
    spec, module, spikes = fixture.synthetic_input(
        topology, STEPS, BATCH, SEED
    )
    return roundtrip(spec, module, spikes, graph=graph, path=out)


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
    _, _, spikes = fixture.synthetic_input(topology, STEPS, BATCH, SEED)
    result = interpret_file(path, spikes)
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

    backend_cli.add_subcommands(subs)
    extract_cli.add_subcommands(subs)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to a deployment subcommand."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-targets",
        description="List deployment targets and run NIR interop.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
