"""Headless ``records`` commands: list, diff, and manifest.

These extend the verify CLI with the Phase 6a tracking surfaces: filtering the
checkpoint registry by its stored metadata, diffing two checkpoints, and
reading one checkpoint's reproducibility manifest. Every command prints JSON;
a missing checkpoint is reported as a plain message with a non-zero exit.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from snn_interpreter.network import model_diff, model_search, model_store

#: Exit status for a command that could not find its checkpoint.
NOT_FOUND = 1
#: Exit status for a command that succeeded.
OK = 0


def search_payload(
    dataset: Optional[str] = None,
    topology: Optional[str] = None,
    coding: Optional[str] = None,
    device: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the matching registry records under a ``models`` key."""
    models = model_search.search_models(
        dataset=dataset, topology=topology, coding=coding,
        device=device, min_accuracy=min_accuracy, name=name,
    )
    return {"models": models}


def diff_payload(left: str, right: str) -> Dict[str, Any]:
    """Return the metadata diff between two checkpoints.

    Raises ``FileNotFoundError`` when either checkpoint is missing, so the
    caller can map it to a non-zero exit.
    """
    return model_diff.checkpoint_diff(left, right)


def manifest_payload(name: str) -> Dict[str, Any]:
    """Return one checkpoint's manifest, with its name attached."""
    if not os.path.exists(model_store.path_for(name)):
        raise FileNotFoundError(f"no checkpoint named {name!r}")
    manifest = model_store.manifest(name)
    return {
        "name": name,
        "available": bool(manifest.get("available")),
        "manifest": manifest,
    }


def _run_list(args: argparse.Namespace) -> int:
    """Print the filtered registry records as JSON."""
    payload = search_payload(
        dataset=args.dataset, topology=args.topology, coding=args.coding,
        device=args.device, min_accuracy=args.min_accuracy, name=args.name,
    )
    print(json.dumps(payload, indent=2))
    return OK


def _run_diff(args: argparse.Namespace) -> int:
    """Print a checkpoint metadata diff, or a not-found message."""
    try:
        payload = diff_payload(args.left, args.right)
    except FileNotFoundError as error:
        print(str(error))
        return NOT_FOUND
    print(json.dumps(payload, indent=2))
    return OK


def _run_manifest(args: argparse.Namespace) -> int:
    """Print a checkpoint manifest, or a not-found message."""
    try:
        payload = manifest_payload(args.name)
    except FileNotFoundError as error:
        print(str(error))
        return NOT_FOUND
    print(json.dumps(payload, indent=2))
    return OK


def _add_list(actions: Any) -> None:
    """Register ``records list`` with its optional filters."""
    listing = actions.add_parser("list", help="filter the registry")
    listing.add_argument("--dataset", default=None)
    listing.add_argument("--topology", default=None)
    listing.add_argument("--coding", default=None)
    listing.add_argument("--device", default=None)
    listing.add_argument(
        "--min-accuracy", dest="min_accuracy", type=float, default=None
    )
    listing.add_argument("--name", default=None)
    listing.set_defaults(handler=_run_list)


def _add_diff(actions: Any) -> None:
    """Register ``records diff A B``."""
    diff = actions.add_parser("diff", help="diff two checkpoints")
    diff.add_argument("left")
    diff.add_argument("right")
    diff.set_defaults(handler=_run_diff)


def _add_manifest(actions: Any) -> None:
    """Register ``records manifest NAME``."""
    man = actions.add_parser("manifest", help="print a checkpoint manifest")
    man.add_argument("name")
    man.set_defaults(handler=_run_manifest)


def add_subcommands(subs: Any) -> None:
    """Register the ``records`` subcommands on ``subs``."""
    records = subs.add_parser(
        "records", help="search and inspect checkpoints"
    )
    actions = records.add_subparsers(dest="records_command", required=True)
    _add_list(actions)
    _add_diff(actions)
    _add_manifest(actions)


def _parser() -> argparse.ArgumentParser:
    """Return a standalone parser with only the records subcommands."""
    parser = argparse.ArgumentParser(
        prog="snn-records",
        description="Search and inspect saved checkpoints.",
    )
    actions = parser.add_subparsers(dest="records_command", required=True)
    _add_list(actions)
    _add_diff(actions)
    _add_manifest(actions)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to the selected records command."""
    args = _parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
