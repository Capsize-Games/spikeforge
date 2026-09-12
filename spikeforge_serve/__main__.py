"""Run the headless inference service: ``python -m spikeforge_serve``."""

import argparse
from typing import Optional, Sequence

import uvicorn

from spikeforge_serve.app import create_app

#: Default bind address and port for the serving API.
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8899


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the bundle path and the optional bind arguments."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-serve",
        description="Serve a .spkf deployment bundle headlessly.",
    )
    parser.add_argument(
        "--bundle",
        required=True,
        help="path to the .spkf deployment bundle to serve",
    )
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
        "--device",
        default="cpu",
        help="torch device the sessions run on (default: cpu)",
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
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Build the app for ``--bundle`` and run it under uvicorn."""
    args = _parse_args(argv)
    app = create_app(
        args.bundle,
        device=args.device,
        metrics_token=args.metrics_token,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
