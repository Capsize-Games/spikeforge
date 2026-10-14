"""Shared entry-validation primitives for the hub's two schemas.

Both the curated catalog (:mod:`spikeforge_hub.entry_checks`) and the
governed registry (:mod:`spikeforge_hub.registry_checks`) validate
string-keyed mappings with an ``id`` label, required text fields, and a
schema projection. Those helpers lived in both modules with identical
bodies; this module holds the single copy, parameterized by the caller's
error type so each schema keeps its typed errors.
"""

import re
from typing import Any, Callable, Dict, Mapping, Sequence

#: Build a schema error from a ``(label, detail)`` pair.
ErrorFactory = Callable[[str, str], Exception]


def text(value: Any) -> bool:
    """Return True when ``value`` is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def label(data: Mapping[str, Any]) -> str:
    """Return the entry id for messages, falling back to ``(unnamed)``."""
    value = data.get("id")
    return value if isinstance(value, str) and value else "(unnamed)"


def check_required(
    data: Mapping[str, Any],
    required: Sequence[str],
    error: ErrorFactory,
) -> None:
    """Reject an entry that omits or blanks a required text field."""
    missing = [key for key in required if not text(data.get(key))]
    if missing:
        raise error(
            label(data), f"missing required fields: {', '.join(missing)}"
        )


def check_id(
    data: Mapping[str, Any],
    pattern: re.Pattern,
    description: str,
    error: ErrorFactory,
) -> None:
    """Reject an id outside ``pattern`` or containing ``..``."""
    entry_id = str(data.get("id", ""))
    if not pattern.match(entry_id):
        raise error(label(data), f"id must match {description}")
    if ".." in entry_id:
        raise error(label(data), "id must not contain '..'")


def project_fields(
    data: Mapping[str, Any], fields: Sequence[str]
) -> Dict[str, Any]:
    """Project ``data`` onto the declared schema ``fields``."""
    return {name: data.get(name) for name in fields}
