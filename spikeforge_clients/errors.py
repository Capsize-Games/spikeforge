"""Typed errors raised by the ``spikeforge-clients`` SDK.

The client never leaks a raw ``HTTPError`` or a JSON-decoding traceback: every
failure is one of a small, named taxonomy so a caller can branch without
parsing prose. :class:`ServiceError` carries the service's own error type, the
HTTP status, and any optional detail, mirroring
``spikeforge_serve.errors.error_body``.
"""

from typing import Any, Mapping, Optional, Tuple


class ClientError(Exception):
    """Base class for every ``spikeforge-clients`` failure."""


class TransportError(ClientError):
    """The request never produced a well-formed HTTP/WebSocket response."""


def describe(payload: Any) -> Tuple[str, str]:
    """Return ``(type, message)`` for a service error ``payload``.

    Handles both the serving taxonomy (``{"error": {"type", "message"}}``) and
    the FastAPI validation shape (``{"detail": "..."}``) so a caller always
    gets a named type and a human message.
    """
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return (
                str(error.get("type", "error")),
                str(error.get("message", "")),
            )
        if payload.get("detail") is not None:
            return ("http_error", str(payload["detail"]))
    return ("http_error", "service returned an error status")


class ServiceError(ClientError):
    """The service answered a non-2xx status with a typed error body."""

    def __init__(
        self,
        status: int,
        error_type: str,
        message: str,
        detail: Any = None,
    ) -> None:
        """Build a typed service error from its status and error body."""
        super().__init__(f"HTTP {status} {error_type}: {message}")
        self.status = status
        self.error_type = error_type
        self.message = message
        self.detail = detail

    @classmethod
    def from_response(cls, status: int, payload: Any) -> "ServiceError":
        """Return the error described by a non-2xx ``payload``."""
        error_type, message = describe(payload)
        return cls(status, error_type, message, _detail(payload))


class StreamError(ClientError):
    """The stream answered with an ``error`` envelope."""

    def __init__(self, error_type: str, message: str) -> None:
        """Build a typed stream error from its envelope."""
        super().__init__(f"{error_type}: {message}")
        self.error_type = error_type
        self.message = message


def _detail(payload: Any) -> Optional[Any]:
    """Return the ``error.detail`` field of ``payload``, when present."""
    if isinstance(payload, Mapping):
        error = payload.get("error")
        if isinstance(error, Mapping):
            return error.get("detail")
    return None
