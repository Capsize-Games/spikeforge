"""Run the FastAPI server with uvicorn: ``python -m server``."""

import argparse
from typing import Optional, Sequence

import uvicorn

from spikeforge.version import add_version_flag

#: Default bind address and port (the single-port dashboard contract on :8877).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8877


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the optional bind arguments, keeping the historical defaults."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-server",
        description="Run the spikeforge dashboard/WebSocket server.",
    )
    add_version_flag(parser)
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"bind address (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"bind port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="disable uvicorn's auto-reload watcher",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Start the dashboard server with uvicorn.

    ``python -m server`` and the ``spikeforge-server`` console script both
    land here; the defaults are the historical ``127.0.0.1:8877`` with
    auto-reload, matching the project's single-port contract.
    """
    args = _parse_args(argv)
    uvicorn.run(
        "server.app:app",
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
    )


if __name__ == "__main__":
    main()
