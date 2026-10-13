"""Account subcommands for the ``spikeforge-hub`` console script.

``login`` stores a pasted personal access token, ``whoami`` and ``quota``
report the signed-in account, and ``push``/``pull`` publish and fetch
community models. Every command prints JSON, mirroring
:mod:`spikeforge_hub.cli`.
"""

import argparse
import getpass
import json
from typing import Any

from spikeforge_hub.client import HubClient, require_client, write_token
from spikeforge_hub.errors import HubError
from spikeforge_hub.transfer import pull_model, push_saved_model


def _print(payload: Any) -> None:
    """Print a payload as indented JSON."""
    print(json.dumps(payload, indent=2))


def _fail(message: str) -> int:
    """Print an error message and return the failure status."""
    print(message)
    return 1


def _read_pat() -> str:
    """Prompt for a personal access token without echoing it."""
    return getpass.getpass("Hub personal access token: ").strip()


def _run_login(args: argparse.Namespace) -> int:
    """Verify a pasted token, then store it for later commands."""
    del args
    pasted = _read_pat()
    if not pasted:
        return _fail("no token pasted; login stores nothing")
    try:
        account = HubClient(token=pasted).me()
    except HubError as error:
        return _fail(f"login refused; nothing stored: {error}")
    _print({"handle": account.get("handle"), "stored": write_token(pasted)})
    return 0


def _run_whoami(args: argparse.Namespace) -> int:
    """Print the signed-in account's profile and storage position."""
    del args
    try:
        account = require_client().me()
    except HubError as error:
        return _fail(str(error))
    _print(account)
    return 0


def _run_quota(args: argparse.Namespace) -> int:
    """Print the signed-in account's storage block."""
    del args
    try:
        account = require_client().me()
    except HubError as error:
        return _fail(str(error))
    storage = account.get("storage")
    if not isinstance(storage, dict):
        return _fail("the server answered without a storage block")
    _print(storage)
    return 0


def _run_push(args: argparse.Namespace) -> int:
    """Build a saved model and publish it through the two-phase upload."""
    try:
        client = require_client()
        report = push_saved_model(
            client,
            args.name,
            args.version,
            args.license,
            slug=args.slug,
            summary=args.summary or "",
            dataset=args.dataset or "",
            dataset_license=args.dataset_license or "",
            dataset_attribution=args.dataset_attribution or "",
        )
    except HubError as error:
        return _fail(str(error))
    _print(report)
    return 0


def _run_pull(args: argparse.Namespace) -> int:
    """Pull a community id and gate on the import verdict, like import."""
    try:
        report = pull_model(args.id)
    except HubError as error:
        return _fail(str(error))
    _print(report)
    return 0 if report["import"]["verdict"]["verdict"] != "incompatible" else 1


def _add_session(subs: Any) -> None:
    """Register the ``login``, ``whoami``, and ``quota`` subcommands."""
    login = subs.add_parser("login", help="store a hub access token")
    login.set_defaults(handler=_run_login)

    whoami = subs.add_parser("whoami", help="show the signed-in account")
    whoami.set_defaults(handler=_run_whoami)

    quota = subs.add_parser("quota", help="show hub storage usage")
    quota.set_defaults(handler=_run_quota)


def _add_push(subs: Any) -> None:
    """Register the ``push`` subcommand."""
    push = subs.add_parser("push", help="publish a saved model")
    push.add_argument("name")
    push.add_argument("--version", required=True)
    push.add_argument("--license", required=True)
    push.add_argument("--slug", default=None)
    push.add_argument("--summary", default=None)
    push.add_argument("--dataset", default=None)
    push.add_argument("--dataset-license", default=None)
    push.add_argument("--dataset-attribution", default=None)
    push.set_defaults(handler=_run_push)


def _add_pull(subs: Any) -> None:
    """Register the ``pull`` subcommand."""
    pull = subs.add_parser("pull", help="pull a community model")
    pull.add_argument("id")
    pull.set_defaults(handler=_run_pull)


def add_subcommands(subs: Any) -> None:
    """Register every account subcommand on ``subs``."""
    _add_session(subs)
    _add_push(subs)
    _add_pull(subs)
