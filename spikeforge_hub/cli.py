"""The ``spikeforge-hub`` console script.

Supports ``list``, ``search``, ``download``, ``inspect``, and ``import``;
mirrors the argparse style of :mod:`spikeforge_targets.cli.target_cli`. Every
command prints JSON. ``import`` exits non-zero when the verdict is
``incompatible``, but still prints the named mismatches, so the command doubles
as a CI gate without hiding *why* an artifact was rejected.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from spikeforge_hub import download as hub_download
from spikeforge_hub import probe
from spikeforge_hub.catalog import get, issues, list_entries, search
from spikeforge_hub.errors import HubArtifactError, HubError
from spikeforge_hub.import_model import import_model
from spikeforge_hub.inspect import inspect_artifact, resolve_path

#: Reason reported when live Hugging Face search is unavailable.
_EXTRA_REASON = "requires the `hub` extra (huggingface_hub)"


def _print(payload: Any) -> None:
    """Print a payload as indented JSON."""
    print(json.dumps(payload, indent=2))


def _fail(message: str) -> int:
    """Print an error message and return the failure status."""
    print(message)
    return 1


def _list_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Return the filtered catalog listing with its validation issues."""
    entries = list_entries(
        framework=args.framework,
        kind=args.kind,
        available=True if args.available else None,
    )
    return {"entries": entries, "issues": issues()}


def _search_payload(args: argparse.Namespace) -> Dict[str, Any]:
    """Return catalog search results plus live-search availability."""
    live = probe.available()
    return {
        "query": args.query,
        "limit": args.limit,
        "available": live,
        "reason": None if live else _EXTRA_REASON,
        "results": search(args.query, args.limit),
    }


def _run_list(args: argparse.Namespace) -> int:
    """Print the catalog listing as JSON."""
    _print(_list_payload(args))
    return 0


def _run_search(args: argparse.Namespace) -> int:
    """Print catalog search results as JSON."""
    _print(_search_payload(args))
    return 0


def _run_download(args: argparse.Namespace) -> int:
    """Download one entry and gate on its verification status."""
    try:
        report = hub_download(args.entry)
    except HubError as error:
        return _fail(str(error))
    _print(report)
    if args.no_verify:
        return 0
    return 1 if report.get("status") == "failed" else 0


def _inspect(entry_id: str) -> Dict[str, Any]:
    """Return the structural report for ``entry_id``'s resolved artifact."""
    entry = get(entry_id)
    if entry is None:
        raise HubArtifactError(entry_id, "unknown catalog entry")
    return inspect_artifact(resolve_path(entry)).to_dict()


def _run_inspect(args: argparse.Namespace) -> int:
    """Print the structural report, or the typed error, plus a status."""
    try:
        report = _inspect(args.entry)
    except HubError as error:
        return _fail(str(error))
    _print(report)
    return 0


def _run_import(args: argparse.Namespace) -> int:
    """Import one entry and gate on the compatibility verdict."""
    try:
        result = import_model(entry_id=args.entry, topology=args.topology)
    except HubError as error:
        return _fail(str(error))
    _print(result)
    return 0 if result["verdict"]["verdict"] != "incompatible" else 1


def _add_listing(subs: Any) -> None:
    """Register the ``list`` and ``search`` subcommands."""
    listing = subs.add_parser("list", help="list curated models")
    listing.add_argument("--framework", default=None)
    listing.add_argument("--kind", default=None)
    listing.add_argument("--available", action="store_true")
    listing.set_defaults(handler=_run_list)

    search_cmd = subs.add_parser("search", help="search the curated catalog")
    search_cmd.add_argument("query")
    search_cmd.add_argument("--limit", type=int, default=20)
    search_cmd.set_defaults(handler=_run_search)


def _add_actions(subs: Any) -> None:
    """Register the ``download``, ``inspect``, and ``import`` subcommands."""
    fetch = subs.add_parser("download", help="download a model into the cache")
    fetch.add_argument("entry")
    fetch.add_argument("--no-verify", action="store_true")
    fetch.set_defaults(handler=_run_download)

    report = subs.add_parser("inspect", help="describe a model's structure")
    report.add_argument("entry")
    report.set_defaults(handler=_run_inspect)

    promote = subs.add_parser("import", help="import a compatible model")
    promote.add_argument("entry")
    promote.add_argument("--topology", default=None)
    promote.set_defaults(handler=_run_import)


def add_subcommands(subs: Any) -> None:
    """Register every hub subcommand on ``subs``."""
    _add_listing(subs)
    _add_actions(subs)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to a hub subcommand."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-hub",
        description="Browse, download, inspect, and import curated models.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
