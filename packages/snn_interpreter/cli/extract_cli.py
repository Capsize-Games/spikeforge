"""Headless ``extract`` command: third-party torch modules via ``nirtorch``.

Registered on the deployment CLI (``snn-targets extract``), this is the ingest
surface for arbitrary ``torch.nn.Module`` objects: the module is loaded from
disk, lifted into NIR by :func:`~snn_interpreter.nir_bridge.extract`, and run
on the independent interpreter. A module ``nirtorch`` cannot map, a tracing
failure, or a missing ``nir`` extra prints the typed reason and exits
non-zero — no stage is ever silently dropped.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

import torch

from snn_interpreter.nir_bridge import extract
from snn_interpreter.nir_bridge.errors import (
    ExtractionError,
    ExtractionExtraMissingError,
    UnsupportedNodeError,
)
from snn_interpreter.nir_bridge.interpreter import NirInterpreter

#: Synthetic run shape used when the module's input size is discoverable.
DEFAULT_STEPS = 8
DEFAULT_BATCH = 2
DEFAULT_SEED = 0

#: Errors ``extract`` turns into a message plus a non-zero exit.
ExtractError = (
    UnsupportedNodeError,
    ExtractionError,
    ExtractionExtraMissingError,
    OSError,
    RuntimeError,
)


def input_features(graph: Any) -> Optional[int]:
    """Return the feature width of ``graph``'s first dense node, if known."""
    for node in graph.nodes.values():
        kind = type(node).__name__
        if kind in ("Linear", "Affine"):
            return int(node.weight.shape[1])
    return None


def _tensor(value: Any) -> List[float]:
    """Return a tensor as a flat list of plain floats."""
    return [float(item) for item in value.reshape(-1)]


def _run(graph: Any, features: int, steps: int, batch: int,
         seed: int) -> Any:
    """Execute ``graph`` over a deterministic flat spike fixture."""
    torch.manual_seed(seed)
    spikes = torch.rand(steps, batch, features)
    return NirInterpreter(graph).run(spikes)


def _payload(
    graph: Any, features: Optional[int], result: Any, steps: int
) -> Dict[str, Any]:
    """Return the JSON-able extraction report for one extracted graph."""
    return {
        "nodes": [
            {"name": name, "kind": type(node).__name__}
            for name, node in graph.nodes.items()
        ],
        "edges": [{"source": src, "target": dst} for src, dst in graph.edges],
        "features": features,
        "steps": steps,
        "runnable": result is not None,
        "readout": None if result is None else _tensor(result.readout),
        "spike_nodes": [] if result is None else sorted(result.spikes),
        "membrane_nodes": [] if result is None else sorted(result.membranes),
    }


def extract_report(
    module: torch.nn.Module,
    steps: int = DEFAULT_STEPS,
    batch: int = DEFAULT_BATCH,
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    """Extract ``module``, run it, and return a JSON-able summary.

    The graph is extracted once and reused for the run, so the report and the
    execution always describe the same artifact.
    """
    graph = extract(module)
    features = input_features(graph)
    if features is None:
        return _payload(graph, features, None, steps)
    result = _run(graph, features, steps, batch, seed)
    return _payload(graph, features, result, steps)


def load_module(path: str) -> torch.nn.Module:
    """Load a serialized ``torch.nn.Module`` from ``path``."""
    loaded = torch.load(path, weights_only=False)
    if not isinstance(loaded, torch.nn.Module):
        raise ExtractionError(f"{path!r} does not contain an nn.Module")
    return loaded


def _run_extract(args: argparse.Namespace) -> int:
    """Print the extraction report, or the typed reason, plus a status."""
    try:
        module = load_module(args.module)
        report = extract_report(module, args.steps, args.batch, args.seed)
    except ExtractError as error:
        print(str(error))
        return 1
    print(json.dumps(report, indent=2))
    return 0


def add_subcommands(subs: Any) -> None:
    """Register the ``extract`` subcommand on ``subs``."""
    extract = subs.add_parser(
        "extract", help="extract a torch module into NIR via nirtorch"
    )
    extract.add_argument("--module", required=True)
    extract.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    extract.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    extract.add_argument("--seed", type=int, default=DEFAULT_SEED)
    extract.set_defaults(handler=_run_extract)


def _parser() -> argparse.ArgumentParser:
    """Return the standalone parser for the ``extract`` command."""
    parser = argparse.ArgumentParser(
        prog="snn-extract",
        description="Extract a torch module into NIR via nirtorch.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the ``extract`` subcommand."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))
