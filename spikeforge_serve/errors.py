"""Map the ``spikeforge.serving`` error taxonomy onto HTTP responses.

The service never invents its own error types: it forwards the typed errors
:mod:`spikeforge.serving.errors` already raises (a missing, malformed,
tampered, or incompatible bundle; a bad encode spec; a bad carried state) and
assigns each an honest HTTP status so a client can react without parsing
prose.
"""

import re
from typing import Any, Dict, Tuple, Type

from spikeforge.serving.errors import (
    BundleCompatibilityError,
    BundleFormatError,
    BundleIntegrityError,
    BundleNotFoundError,
    EncodeSpecError,
    ServingError,
    StateError,
)

#: ``(error class, status)`` pairs, most specific first.
_STATUS: Tuple[Tuple[Type[ServingError], int], ...] = (
    (BundleNotFoundError, 404),
    (BundleCompatibilityError, 409),
    (BundleIntegrityError, 422),
    (BundleFormatError, 422),
    (EncodeSpecError, 422),
    (StateError, 409),
)


def status_for(error: ServingError) -> int:
    """Return the HTTP status code that fits ``error``."""
    for error_type, status in _STATUS:
        if isinstance(error, error_type):
            return status
    return 500


def error_type(error: ServingError) -> str:
    """Return ``error``'s snake_case type name for the JSON payload."""
    name = type(error).__name__
    if name.endswith("Error"):
        name = name[: -len("Error")]
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def error_body(error: ServingError) -> Dict[str, Any]:
    """Return a JSON-ready body describing ``error``."""
    payload: Dict[str, Any] = {
        "type": error_type(error),
        "message": str(error),
    }
    for attribute in ("detail", "path", "entry"):
        value = getattr(error, attribute, None)
        if value is not None:
            payload[attribute] = value
    return {"error": payload}
