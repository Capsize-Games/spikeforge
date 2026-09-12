"""Typed liveness and readiness results for the service probes."""

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from spikeforge_clients.errors import describe


def _text(payload: Any, key: str, default: str) -> str:
    """Return ``payload[key]`` as text, or ``default`` when absent."""
    if isinstance(payload, Mapping) and payload.get(key) is not None:
        return str(payload[key])
    return default


@dataclass(frozen=True)
class Health:
    """The ``/health`` liveness payload."""

    status: str

    @classmethod
    def from_json(cls, payload: Any) -> "Health":
        """Decode the liveness body."""
        return cls(status=_text(payload, "status", "ok"))


@dataclass(frozen=True)
class Readiness:
    """The ``/ready`` result: ready, or a named reason it is not."""

    ready: bool
    status: str
    reason: Optional[str] = None

    @classmethod
    def from_response(cls, status: int, payload: Any) -> "Readiness":
        """Decode a readiness response, mapping an error body to a reason."""
        if status < 400:
            return cls(True, _text(payload, "status", "ready"))
        error_type, message = describe(payload)
        return cls(False, error_type, message)
