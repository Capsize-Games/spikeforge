"""The ``spikeforge-clients`` console script.

Wraps :class:`~spikeforge_clients.client.ServeClient` for ``info``,
``predict`` (frames from a JSON or ``.npy`` file), and ``stream`` (frames read
one JSON document per line from stdin). Every command prints JSON; a typed
client failure prints a message and exits 1, mirroring the repository's CLI
style.
"""

import argparse
import json
import sys
from typing import Any, Iterator, List, Optional, Sequence, TextIO

from spikeforge_clients.client import ServeClient
from spikeforge_clients.config import DEFAULT_BASE_URL, DEFAULT_TIMEOUT
from spikeforge_clients.errors import ClientError


def _parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    """Parse the connection options and the subcommand."""
    parser = argparse.ArgumentParser(
        prog="spikeforge-clients",
        description="Call a spikeforge-serve inference service.",
    )
    _add_connection(parser)
    subs = parser.add_subparsers(dest="command", required=True)
    _add_info(subs)
    _add_predict(subs)
    _add_stream(subs)
    return parser.parse_args(argv)


def _add_connection(parser: argparse.ArgumentParser) -> None:
    """Register the shared connection options."""
    parser.add_argument("--url", default=DEFAULT_BASE_URL, help="service URL")
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT, help="seconds"
    )
    parser.add_argument(
        "--token", default=None, help="optional bearer token"
    )


def _add_info(subs: Any) -> None:
    """Register the ``info`` subcommand."""
    info = subs.add_parser("info", help="print bundle metadata as JSON")
    info.set_defaults(handler=_run_info)


def _add_predict(subs: Any) -> None:
    """Register the ``predict`` subcommand."""
    predict = subs.add_parser("predict", help="predict from a file of frames")
    predict.add_argument("--file", required=True, help="JSON or .npy file")
    predict.add_argument("--encoded", action="store_true")
    predict.add_argument("--session-id", dest="session_id", default=None)
    predict.set_defaults(handler=_run_predict)


def _add_stream(subs: Any) -> None:
    """Register the ``stream`` subcommand."""
    stream = subs.add_parser(
        "stream", help="stream stdin frames as JSON lines"
    )
    stream.add_argument("--encoded", action="store_true")
    stream.add_argument("--session-id", dest="session_id", default=None)
    stream.set_defaults(handler=_run_stream)


def _client(args: argparse.Namespace) -> ServeClient:
    """Build a client for the parsed connection options."""
    return ServeClient(
        base_url=args.url, timeout=args.timeout, token=args.token
    )


def _load_frames(path: str) -> List[Any]:
    """Load frames from a ``.npy`` array or a JSON document."""
    if path.endswith(".npy"):
        return _normalize(_load_npy(path))
    try:
        with open(path, encoding="utf-8") as handle:
            return _normalize(json.load(handle))
    except ValueError as error:
        raise ClientError(f"{path} is not valid JSON: {error}") from error


def _load_npy(path: str) -> Any:
    """Load a numpy array and return its nested Python values."""
    try:
        import numpy as np
    except ImportError as error:
        raise ClientError("reading .npy frames requires numpy") from error
    return np.load(path).tolist()


def _normalize(document: Any) -> List[Any]:
    """Return ``document`` as a non-empty list of frames."""
    if isinstance(document, dict):
        document = document.get("frames")
    if not isinstance(document, list) or not document:
        raise ClientError("frames must be a non-empty JSON list")
    return document


def _iter_frames(stream: TextIO) -> Iterator[Any]:
    """Yield one frame per non-empty line of ``stream`` (JSON per line)."""
    for line in stream:
        text = line.strip()
        if text:
            yield _parse_json(text)


def _parse_json(text: str) -> Any:
    """Return ``text`` parsed as JSON, or raise a typed ``ClientError``."""
    try:
        return json.loads(text)
    except ValueError as error:
        raise ClientError(f"invalid JSON frame: {error}") from error


def _run_info(args: argparse.Namespace) -> int:
    """Print the bundle metadata as pretty JSON."""
    payload = _client(args).bundle_info().to_dict()
    print(json.dumps(payload, indent=2))
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    """Predict from a file of frames and print the response as JSON."""
    frames = _load_frames(args.file)
    response = _client(args).predict(
        frames, encoded=args.encoded, session_id=args.session_id
    )
    print(json.dumps(response.to_dict(), indent=2))
    return 0


def _run_stream(args: argparse.Namespace) -> int:
    """Stream frames from stdin and print one JSON reply per line."""
    frames = _iter_frames(sys.stdin)
    events = _client(args).stream(
        frames, encoded=args.encoded, session_id=args.session_id
    )
    for event in events:
        print(json.dumps(event.to_dict()))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """Parse ``argv`` and dispatch to a client subcommand."""
    args = _parse_args(argv)
    try:
        return int(args.handler(args))
    except (ClientError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
