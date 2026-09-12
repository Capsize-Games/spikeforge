"""The ``spikeforge-registry`` console script.

Governs the signable promotion registry introduced by PT-W7. Supports
``list``, ``validate``, ``verify``, and ``promote``; mirrors the argparse style
of :mod:`spikeforge_hub.cli` and prints JSON on every command. ``validate`` and
``verify`` exit non-zero on any typed refusal, so the command doubles as a CI
gate without hiding *why* an entry was refused — the same contract
``spikeforge-hub import`` uses for compatibility.

The script is owned by the hub distribution (``spikeforge-registry``), which is
the plan's "extend hub" option; no separate import root is introduced.
"""

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from spikeforge_hub import registry
from spikeforge_hub.registry import Registry, RegistryEntry
from spikeforge_hub.registry_errors import RegistryError

#: Environment variable read when ``--key`` is not supplied.
KEY_ENV = "SPIKEFORGE_REGISTRY_KEY"


def _print(payload: Any) -> None:
    """Print a payload as indented JSON."""
    print(json.dumps(payload, indent=2))


def _fail(message: str) -> int:
    """Print an error message and return the failure status."""
    print(message)
    return 1


def _key(args: argparse.Namespace) -> str:
    """Return the signing key from ``--key`` or the environment."""
    value = args.key or os.environ.get(KEY_ENV)
    if not value:
        raise RegistryError(
            f"no signing key: pass --key or set {KEY_ENV}"
        )
    return value


def _run_list(args: argparse.Namespace) -> int:
    """Print the ordered registry with its issues and flagged entries."""
    try:
        loaded = Registry.load(args.registry)
    except RegistryError as error:
        return _fail(str(error))
    _print(
        {
            "registry": args.registry or registry.default_registry_path(),
            "entries": [item.to_dict() for item in loaded.ordered()],
            "flagged": loaded.flagged(),
            "issues": list(loaded.issues),
        }
    )
    return 1 if loaded.issues else 0


def _run_validate(args: argparse.Namespace) -> int:
    """Validate the registry and curated catalog, then gate on issues."""
    payload: Dict[str, Any] = {"issues": []}
    if not args.catalog_only:
        try:
            loaded = Registry.load(args.registry)
        except RegistryError as error:
            return _fail(str(error))
        payload["registry"] = loaded.to_dict()
        payload["issues"].extend(loaded.issues)
    if not args.registry_only:
        entries, problems = registry.validate_catalog(args.catalog)
        payload["catalog_entries"] = len(entries)
        payload["issues"].extend(problems)
    payload["ok"] = not payload["issues"]
    _print(payload)
    return 0 if payload["ok"] else 1


def _find(loaded: Registry, entry_id: str) -> RegistryEntry:
    """Return the governed entry, or raise a typed error naming it."""
    entry = loaded.get(entry_id)
    if entry is None:
        raise RegistryError(f"unknown registry entry {entry_id!r}")
    return entry


def _run_verify(args: argparse.Namespace) -> int:
    """Verify one entry's signature and artifact checksum."""
    try:
        loaded = Registry.load(args.registry)
        entry = _find(loaded, args.entry)
        registry.verify_entry(entry, _key(args))
        artifact = registry.verify_artifact(entry)
    except RegistryError as error:
        return _fail(str(error))
    _print(
        {
            "id": entry.id,
            "stage": entry.stage,
            "status": entry.status,
            "signature": "verified",
            "artifact": artifact,
            "lineage": registry.lineage_report(entry),
        }
    )
    return 0


def _run_promote(args: argparse.Namespace) -> int:
    """Promote one entry, recording the approver and a fresh signature."""
    try:
        loaded = Registry.load(args.registry)
        entry = _find(loaded, args.entry)
        promoted = registry.promote(
            entry, args.to, args.approver, _key(args), at=args.at
        )
        updated = loaded.add(promoted)
        path = updated.save(args.registry)
    except RegistryError as error:
        return _fail(str(error))
    _print(
        {
            "promoted": True,
            "registry": path,
            "entry": promoted.to_dict(),
            "signature": "signed",
            "lineage": registry.lineage_report(promoted),
        }
    )
    return 0


def _add_registry(subs: Any) -> None:
    """Register the ``list`` and ``validate`` subcommands."""
    listing = subs.add_parser("list", help="list the governed registry")
    listing.add_argument("--registry", default=None)
    listing.set_defaults(handler=_run_list)

    check = subs.add_parser("validate", help="validate registry and catalog")
    check.add_argument("--registry", default=None)
    check.add_argument("--catalog", default=None)
    check.add_argument("--registry-only", action="store_true")
    check.add_argument("--catalog-only", action="store_true")
    check.set_defaults(handler=_run_validate)


def _add_actions(subs: Any) -> None:
    """Register the ``verify`` and ``promote`` subcommands."""
    verify = subs.add_parser("verify", help="verify one entry and signature")
    verify.add_argument("entry")
    verify.add_argument("--registry", default=None)
    verify.add_argument("--key", default=None)
    verify.set_defaults(handler=_run_verify)

    promote = subs.add_parser("promote", help="advance an entry a stage")
    promote.add_argument("entry")
    promote.add_argument("--to", required=True, choices=list(registry.STAGES))
    promote.add_argument("--approver", required=True)
    promote.add_argument("--registry", default=None)
    promote.add_argument("--key", default=None)
    promote.add_argument("--at", type=float, default=None)
    promote.set_defaults(handler=_run_promote)


def add_subcommands(subs: Any) -> None:
    """Register every registry subcommand on ``subs``."""
    _add_registry(subs)
    _add_actions(subs)


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to a registry subcommand."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-registry",
        description="Govern the dev/staging/prod promotion registry.",
    )
    subs = parser.add_subparsers(dest="command", required=True)
    add_subcommands(subs)
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
