"""One-shot, non-HTTP inference: the Unix-pipe path through a module.

``spikeforge-serve run`` is the same input -> output contract as
:mod:`spikeforge_serve.app`'s ``/v1/predict`` route, minus the server: read
one JSON request from stdin (or a file), run it through
:class:`~spikeforge_serve.service.ServingService`, print one JSON response.
That symmetry is deliberate -- a module behaves identically whether it is
served over HTTP or run as a standalone command.
"""

import json
import sys
from typing import IO, Any, Dict, Optional

from spikeforge_serve import modules
from spikeforge_serve.payloads import (
    encoded_flag,
    frames_from,
    prediction_json,
    session_id_from,
)
from spikeforge_serve.service import ServingService


def resolve_bundle(name_or_path: str) -> str:
    """Return a bundle path for an installed module name or a literal path.

    An installed module name is tried first so an unqualified name never
    accidentally resolves to a same-named file in the working directory;
    anything not installed falls back to being a path, so running a
    ``.spkf`` straight off disk works without installing it first.
    """
    try:
        return modules.bundle_path_for(name_or_path)
    except FileNotFoundError:
        return name_or_path


def _read_request(source: IO[str]) -> Dict[str, Any]:
    """Parse the single JSON request object from ``source``."""
    text = source.read()
    if not text.strip():
        raise ValueError("no input: expected a JSON object on stdin or --file")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("input must be a JSON object")
    return payload


def run(
    name_or_path: str,
    input_file: Optional[str] = None,
    device: str = "cpu",
    out: IO[str] = sys.stdout,
) -> int:
    """Run one request through ``name_or_path``; print the result as JSON.

    Returns a process exit code (0 on success) so the CLI entry point can
    forward it directly.
    """
    payload = _load_payload(input_file)
    service = ServingService(resolve_bundle(name_or_path), device=device)
    result = _run_request(service, payload)
    json.dump(result, out)
    out.write("\n")
    return 0


def _load_payload(input_file: Optional[str]) -> Dict[str, Any]:
    """Read the request JSON from ``input_file`` or stdin."""
    if input_file is not None:
        with open(input_file, encoding="utf-8") as handle:
            return _read_request(handle)
    return _read_request(sys.stdin)


def _run_request(
    service: ServingService, payload: Dict[str, Any]
) -> Dict[str, Any]:
    """Apply an optional reset, then predict and shape the JSON result."""
    if payload.get("reset"):
        service.reset(session_id_from(payload))
    frames = frames_from(payload)
    predictions = service.predict(
        frames,
        session_id=session_id_from(payload),
        encoded=encoded_flag(payload),
    )
    return {"predictions": [prediction_json(p) for p in predictions]}
