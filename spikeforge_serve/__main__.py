"""The ``spikeforge-serve`` CLI: serve a bundle, or run/install it as a module.

Subcommands:

- ``serve``     start the HTTP/WebSocket inference service (the original
                behaviour of this entry point).
- ``run``       one-shot inference: read a JSON request from stdin (or
                ``--file``), print the JSON response, exit. No server.
- ``install``   register a ``.spkf`` bundle as a named module: a copy under
                ``~/.local/share/spikeforge/modules/<name>/`` plus an
                executable ``<name>`` wrapper on ``~/.local/bin``.
- ``uninstall`` remove an installed module and its wrapper script.
- ``list``      show every installed module.
"""

import argparse
import json
import sys
from typing import Optional, Sequence

import uvicorn

from spikeforge.version import add_version_flag
from spikeforge_serve import modules
from spikeforge_serve.app import create_app
from spikeforge_serve.module_runner import run as run_module

#: Default bind address and port for the serving API.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8899


def _add_serve_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``serve``: start the HTTP/WebSocket inference service."""
    parser = subparsers.add_parser(
        "serve", help="serve a .spkf bundle over HTTP/WebSocket"
    )
    parser.add_argument(
        "--bundle", required=True, help="path to the .spkf deployment bundle"
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device the sessions run on"
    )
    parser.add_argument(
        "--metrics-token",
        dest="metrics_token",
        default=None,
        help=(
            "bearer token required by /metrics (also read from "
            "SPIKEFORGE_SERVE_METRICS_TOKEN; unset leaves /metrics open)"
        ),
    )


def _add_run_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``run``: one-shot inference through a module or a bundle path."""
    parser = subparsers.add_parser(
        "run", help="run one request through an installed module or a .spkf"
    )
    parser.add_argument(
        "name_or_path",
        help="an installed module name, or a path to a .spkf bundle",
    )
    parser.add_argument(
        "--file",
        dest="input_file",
        default=None,
        help="read the JSON request from this file instead of stdin",
    )
    parser.add_argument(
        "--device", default="cpu", help="torch device to run on (default: cpu)"
    )


def _add_install_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``install``: register a bundle as a named module."""
    parser = subparsers.add_parser(
        "install", help="install a .spkf bundle as a named module"
    )
    parser.add_argument("bundle", help="path to the .spkf bundle to install")
    parser.add_argument(
        "--name",
        default=None,
        help="module name (default: the bundle's file stem)",
    )
    parser.add_argument(
        "--no-wrapper",
        dest="make_wrapper",
        action="store_false",
        help="skip writing an executable wrapper script",
    )


def _add_uninstall_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``uninstall``: remove an installed module."""
    parser = subparsers.add_parser(
        "uninstall", help="remove an installed module"
    )
    parser.add_argument("name", help="the module name to remove")


def _add_list_parser(subparsers: "argparse._SubParsersAction") -> None:
    """Add ``list``: show every installed module."""
    subparsers.add_parser("list", help="list installed modules")


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse a subcommand and its arguments."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-serve",
        description="Serve, run, and install .spkf deployment bundles.",
    )
    add_version_flag(parser)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_serve_parser(subparsers)
    _add_run_parser(subparsers)
    _add_install_parser(subparsers)
    _add_uninstall_parser(subparsers)
    _add_list_parser(subparsers)
    return parser.parse_args(argv)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Build the app for ``--bundle`` and run it under uvicorn."""
    app = create_app(
        args.bundle, device=args.device, metrics_token=args.metrics_token
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one request through a module or bundle path."""
    return run_module(
        args.name_or_path, input_file=args.input_file, device=args.device
    )


def _cmd_install(args: argparse.Namespace) -> int:
    """Install a bundle as a named module and report where it landed."""
    target = modules.install(
        args.bundle, name=args.name, make_wrapper=args.make_wrapper
    )
    descriptor = json.loads((target / "module.json").read_text())
    name = descriptor["name"]
    print(f"installed {name!r} -> {target}")
    if args.make_wrapper:
        print(
            f"run it directly with `{name}` if {modules.BIN_HOME} is on PATH, "
            f"or `spikeforge-serve run {name}`"
        )
    return 0


def _cmd_uninstall(args: argparse.Namespace) -> int:
    """Remove an installed module; report whether there was one."""
    if modules.uninstall(args.name):
        print(f"removed {args.name!r}")
        return 0
    print(f"no module named {args.name!r}", file=sys.stderr)
    return 1


def _cmd_list(_args: argparse.Namespace) -> int:
    """Print every installed module, one per line."""
    installed = modules.list_modules()
    if not installed:
        print("no modules installed")
        return 0
    for entry in installed:
        classes = entry.get("num_classes")
        topology = entry.get("topology")
        print(f"{entry['name']}\t{topology}\t{classes} classes")
    return 0


_COMMANDS = {
    "serve": _cmd_serve,
    "run": _cmd_run,
    "install": _cmd_install,
    "uninstall": _cmd_uninstall,
    "list": _cmd_list,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Dispatch to the parsed subcommand; return its exit code."""
    args = _parse_args(argv)
    return _COMMANDS[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
