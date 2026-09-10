"""Headless ``export`` and ``validate`` commands for the NIR bridge.

Run as ``python -m snn_interpreter.cli.verify export --topology conv_net`` or
``python -m snn_interpreter.cli.verify validate --topology conv_net``. The
``validate`` command exits non-zero when the report falls outside tolerance,
so it works as a CI gate.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from snn_interpreter.data.datasets import dataset_info
from snn_interpreter.data.sample_source import SampleSource
from snn_interpreter.encoding.spike_encoder import SpikeEncoder
from snn_interpreter.nir_bridge import graph_summary, to_nir, validate
from snn_interpreter.nir_bridge.validation_report import ValidationReport
from snn_interpreter.simulator import input_shape
from snn_interpreter.topology.registry import build_topology
from snn_interpreter.topology.spec import TopologySpec
from snn_interpreter.topology.stage_module import StageModule

_Input = Tuple[TopologySpec, StageModule, torch.Tensor]


def export_summary(topology: str) -> Dict[str, Any]:
    """Build ``topology``, export it, and return its JSON-able summary."""
    spec, module = build_topology(topology)
    return graph_summary(to_nir(spec, module))


def sample_input(
    topology: str,
    dataset: str,
    sample: int = 0,
    steps: int = 10,
    seed: int = 0,
) -> _Input:
    """Build ``topology`` and one encoded, correctly-shaped spike sample."""
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
    """Print the graph summary, or write it to ``--out``."""
    text = json.dumps(export_summary(args.topology), indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    """Build a real sample, print the report, and return its exit status."""
    spec, module, spikes = sample_input(
        args.topology, args.dataset, args.sample, args.steps, args.seed
    )
    report = validate_report(spec, module, spikes)
    print(json.dumps(report, indent=2))
    return exit_code(report)


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the verify CLI."""
    parser = argparse.ArgumentParser(
        prog="snn-verify",
        description="Export and validate topology NIR graphs.",
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
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected subcommand."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    sys.exit(main())
