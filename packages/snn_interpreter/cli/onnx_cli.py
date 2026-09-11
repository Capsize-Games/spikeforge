"""Headless ONNX interop commands: ``onnx-export``, ``onnx-import``.

Registered on the verify CLI (``snn-verify onnx-export``), these expose the
optional :mod:`snn_interpreter.onnx_bridge` to the command line. A missing
``onnx`` extra or an unmappable op prints the typed reason and exits non-zero,
so the commands stay honest CI gates rather than raising a traceback.
"""

import argparse
import json
from typing import Any, List, Optional

from snn_interpreter.onnx_bridge import export, import_onnx
from snn_interpreter.onnx_bridge import roundtrip as roundtrip_report
from snn_interpreter.onnx_bridge.errors import OnnxBridgeError


def _run_export(args: argparse.Namespace) -> int:
    """Print the export report, or the typed reason, plus a status."""
    try:
        report = export.export_topology(args.topology, args.out)
    except OnnxBridgeError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _run_import(args: argparse.Namespace) -> int:
    """Print the import report, or the typed reason, plus a status."""
    try:
        report = import_onnx.import_report(args.file)
    except OnnxBridgeError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    return 0


def _run_roundtrip(args: argparse.Namespace) -> int:
    """Print the ONNX round-trip report and return its fidelity status."""
    try:
        report = roundtrip_report(args.topology, args.out)
    except OnnxBridgeError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    return 0 if report["identical"] else 1


def add_subcommands(subs: Any) -> None:
    """Register the ONNX interop subcommands on ``subs``."""
    draw = subs.add_parser("onnx-export", help="export a topology to ONNX")
    draw.add_argument("--topology", default="conv_net")
    draw.add_argument("--out", required=True)
    draw.set_defaults(handler=_run_export)

    load = subs.add_parser("onnx-import", help="import an ONNX graph")
    load.add_argument("--file", required=True)
    load.set_defaults(handler=_run_import)

    trip = subs.add_parser(
        "onnx-roundtrip", help="export and re-import via ONNX"
    )
    trip.add_argument("--topology", default="conv_net")
    trip.add_argument("--out", default=None)
    trip.set_defaults(handler=_run_roundtrip)


def _parser() -> argparse.ArgumentParser:
    """Return the standalone parser for the ONNX interop commands."""
    parser = argparse.ArgumentParser(
        prog="snn-onnx",
        description="Export and import topologies through ONNX.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected ONNX subcommand."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))
