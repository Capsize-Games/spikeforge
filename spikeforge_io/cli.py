"""The ``spikeforge-io`` console script.

A small inspection/replay surface over the adapters: ``formats`` lists the
registered suffixes, ``info`` describes a recorded stream, and ``replay``
prints the ``/v1/stream`` messages an adapter would send (a dry run, so it can
be used in CI without a running service). Mirrors the argparse style of
:mod:`spikeforge_hub.cli`.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

import torch

from spikeforge_io.adapters import FACTORIES, adapter_for
from spikeforge_io.errors import IoError
from spikeforge_io.replay import stream_messages


def _print(payload: Any) -> None:
    """Print a payload as indented JSON."""
    print(json.dumps(payload, indent=2, default=_default))


def _default(value: Any) -> Any:
    """Coerce a non-JSON value (a tensor shape) for printing."""
    if isinstance(value, torch.Tensor):
        return value.tolist()
    return str(value)


def _fail(message: str) -> int:
    """Print an error message and return the failure status."""
    print(message)
    return 1


def _run_formats(_args: argparse.Namespace) -> int:
    """Print the registered adapter suffixes."""
    _print({"formats": sorted(FACTORIES)})
    return 0


def _run_info(args: argparse.Namespace) -> int:
    """Describe a recorded stream: shape, channels, and value range."""
    try:
        adapter = adapter_for(
            args.path,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
        stream = adapter.stream()
    except IoError as error:
        return _fail(str(error))
    _print(
        {
            "source": adapter.source,
            "samples": int(stream.size(0)),
            "channels": int(stream.size(1)),
            "min": float(stream.min().item()) if stream.numel() else None,
            "max": float(stream.max().item()) if stream.numel() else None,
            "mean": float(stream.mean().item()) if stream.numel() else None,
        }
    )
    return 0


def _run_replay(args: argparse.Namespace) -> int:
    """Print the ``/v1/stream`` messages an adapter would send (a dry run)."""
    try:
        adapter = adapter_for(
            args.path,
            delimiter=args.delimiter,
            has_header=not args.no_header,
        )
        spec = None
        if args.length is not None:
            from spikeforge_io.windowing import fit_window_spec

            spec = fit_window_spec(
                adapter.stream(), int(args.length), int(args.stride or 1)
            )
        messages: List[Dict[str, Any]] = list(
            stream_messages(
                adapter,
                window_spec=spec,
                encoded=args.encoded,
                session_id=args.session_id,
                reset=args.reset,
            )
        )
    except IoError as error:
        return _fail(str(error))
    _print({"source": adapter.source, "messages": messages})
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to an I/O subcommand."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-io",
        description="Inspect and replay recorded numeric streams.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    formats = subs.add_parser("formats", help="list adapter formats")
    formats.set_defaults(handler=_run_formats)

    info = subs.add_parser("info", help="describe a recorded stream")
    _add_source(info)
    info.set_defaults(handler=_run_info)

    replay = subs.add_parser("replay", help="print /v1/stream messages")
    _add_source(replay)
    replay.add_argument("--length", type=int, default=None)
    replay.add_argument("--stride", type=int, default=None)
    replay.add_argument("--encoded", action="store_true")
    replay.add_argument("--reset", action="store_true")
    replay.add_argument("--session-id", dest="session_id", default=None)
    replay.set_defaults(handler=_run_replay)

    args = parser.parse_args(argv)
    return int(args.handler(args))


def _add_source(parser: argparse.ArgumentParser) -> None:
    """Add the shared source arguments to ``parser``."""
    parser.add_argument("path")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--no-header", action="store_true")


if __name__ == "__main__":
    raise SystemExit(main())
