"""Headless NIR verify commands: ``export``, ``validate`` and Phase 5c.

Run as ``python -m spikeforge.cli.verify export --topology conv_net`` or
``python -m spikeforge.cli.verify validate --topology conv_net``. The
``validate`` command exits non-zero when the report falls outside tolerance,
so it works as a CI gate. The deployment subcommands (``targets``, ``deploy``,
``roundtrip``, ``ingest``) are registered from
:mod:`spikeforge_targets.cli.target_cli` when that optional package is
installed, and the checkpoint-tracking subcommands (``records
list|diff|manifest``) from :mod:`.records_cli`.

``spikeforge-targets`` is not a hard dependency of the core distribution, so
this module imports it defensively: without it, ``--help`` and the
``export``/``validate``/``records``/``onnx-*`` subcommands still work, and
the deployment subcommands are simply absent from the subcommand list (the
``--help`` description names the install that brings them back).
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from spikeforge.cli import fixture, onnx_cli, records_cli
from spikeforge.data.datasets import dataset_info
from spikeforge.data.sample_source import SampleSource
from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.nir_bridge import graph_summary, to_nir, validate
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.validation_report import ValidationReport
from spikeforge.simulator import input_shape
from spikeforge.topology import registry
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

try:
    from spikeforge_targets.cli import target_cli
except ImportError:
    target_cli = None

_Input = Tuple[TopologySpec, StageModule, torch.Tensor]


def export_summary(topology: str) -> Dict[str, Any]:
    """Build ``topology``, export it, and return its JSON-able summary."""
    spec, module = build_topology(topology)
    return graph_summary(to_nir(spec, module))


def _sequence_input(topology: str, steps: int, seed: int) -> _Input:
    """Build a sequence topology and one deterministic sequence fixture."""
    return fixture.sequence_input(topology, steps, 1, seed)


def sample_input(
    topology: str,
    dataset: str,
    sample: int = 0,
    steps: int = 10,
    seed: int = 0,
) -> _Input:
    """Build ``topology`` and one encoded, correctly-shaped spike sample.

    Sequence topologies receive a ``[T, B, L, D]`` token fixture (or
    ``[T, B, L]`` integer tokens for an ``embedding`` entry) instead of an
    encoded image, so the same command validates the sequence data path.
    """
    if registry.is_sequence_topology(topology):
        return _sequence_input(topology, steps, seed)
    num_classes, _ = dataset_info(dataset)
    spec, module = build_topology(topology, {"num_classes": num_classes})
    torch.manual_seed(seed)
    image = SampleSource(dataset=dataset).image(sample)
    encoder = SpikeEncoder(coding="rate", num_steps=steps)
    spikes = input_shape.to_input_shape(encoder.encode_image(image), spec)
    return spec, module, spikes


def validate_report(
    spec: TopologySpec,
    module: StageModule,
    spikes: torch.Tensor,
    graph: Optional[Any] = None,
) -> ValidationReport:
    """Return the validation report for a built module and spike input."""
    return validate(spec, module, spikes, graph=graph)


def exit_code(report: ValidationReport) -> int:
    """Return the process exit status for a validation report."""
    return 0 if report["within_tolerance"] else 1


def _run_export(args: argparse.Namespace) -> int:
    """Print the graph summary, or write it to ``--out``.

    A topology with an unexportable stage prints the typed error naming the
    stage and exits non-zero, so the command stays a usable CI gate.
    """
    try:
        summary = export_summary(args.topology)
    except UnsupportedStageError as exc:
        print(json.dumps({"error": str(exc), "kind": exc.kind}, indent=2))
        return 1
    text = json.dumps(summary, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    """Build a real sample, print the report, and return its exit status.

    An unexportable topology cannot be validated, so the typed error naming
    the stage is printed and the command exits non-zero instead of raising.
    """
    spec, module, spikes = sample_input(
        args.topology, args.dataset, args.sample, args.steps, args.seed
    )
    try:
        report = validate_report(spec, module, spikes)
    except UnsupportedStageError as exc:
        print(json.dumps({"error": str(exc), "kind": exc.kind}, indent=2))
        return 1
    print(json.dumps(report, indent=2))
    return exit_code(report)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the verify CLI."""
    description = "Export and validate topology NIR graphs."
    if target_cli is None:
        description += (
            " (deployment subcommands -- targets, deploy, roundtrip, "
            "test-deploy, ingest -- need: pip install spikeforge-targets)"
        )
    parser = argparse.ArgumentParser(
        prog="spikeforge-verify",
        description=description,
    )
    subs = parser.add_subparsers(dest="command", required=True)
    export = subs.add_parser("export", help="export a topology to NIR")
    export.add_argument("--topology", default="conv_net")
    export.add_argument("--out", default=None)
    export.set_defaults(handler=_run_export)
    check = subs.add_parser("validate", help="validate an NIR graph")
    check.add_argument("--topology", default="conv_net")
    check.add_argument("--dataset", default="mnist")
    check.add_argument("--sample", type=int, default=0)
    check.add_argument("--steps", type=int, default=10)
    check.add_argument("--seed", type=int, default=0)
    check.set_defaults(handler=_run_validate)
    if target_cli is not None:
        target_cli.add_subcommands(subs)
    records_cli.add_subcommands(subs)
    onnx_cli.add_subcommands(subs)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
