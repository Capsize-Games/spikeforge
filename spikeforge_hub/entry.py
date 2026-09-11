"""Typed metadata for a curated model-hub catalog entry.

Each entry is validated against the schema in ``models.json`` before it is
usable, so a malformed record is reported as a typed
:class:`~spikeforge_hub.errors.HubCatalogError` instead of being
silently skipped. Unknown frameworks, kinds, and sources are rejected by
name, honouring the project's honesty rule.
"""

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

from spikeforge_hub.errors import HubCatalogError

#: Frameworks the curated catalog may declare.
FRAMEWORKS: Tuple[str, ...] = (
    "snntorch",
    "nir",
    "spikingjelly",
    "norse",
    "lava",
    "hf",
)
#: Artifact kinds an entry may describe.
KINDS: Tuple[str, ...] = ("nir_graph", "state_dict", "framework_weights")
#: Where an entry's artifact comes from.
SOURCES: Tuple[str, ...] = ("bundled", "url", "hf_repo")

#: Explicit marker for a candidate whose upstream license is not verified.
#:
#: An entry carrying this marker is allowed to load only so it can be reported
#: as *not ready* (see :func:`spikeforge_hub.catalog.availability`); it is
#: never presented as an available, ready-to-use artifact.
UNVERIFIED_CANDIDATE = "unverified-candidate"
#: One SPDX-style id (no whitespace), optionally joined by SPDX operators.
_LICENSE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9.+-]*"
    r"(?:\s+(?:AND|OR|WITH)\s+[A-Za-z0-9][A-Za-z0-9.+-]*)*$"
)

_REQUIRED: Tuple[str, ...] = (
    "id",
    "name",
    "framework",
    "kind",
    "source",
    "license",
    "notes",
)
_FIELDS: Tuple[str, ...] = _REQUIRED + (
    "url",
    "hf_repo",
    "sha256",
    "size_bytes",
    "topology",
    "input_shape",
)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


@dataclass(frozen=True)
class HubEntry:
    """A curated catalog entry: where a model comes from and what it is."""

    id: str
    name: str
    framework: str
    kind: str
    source: str
    license: str
    notes: str
    url: Optional[str] = None
    hf_repo: Optional[str] = None
    sha256: Optional[str] = None
    size_bytes: Optional[int] = None
    topology: Optional[str] = None
    input_shape: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HubEntry":
        """Validate ``data`` and build an entry, raising on any problem."""
        _check_required(data)
        _check_id(data)
        _check_allowed(data)
        _check_license(data)
        _check_source(data)
        return cls(**_fields(data))

    def to_dict(self) -> Dict[str, Any]:
        """Return the entry as a JSON-able mapping in schema field order."""
        values = asdict(self)
        return {name: values[name] for name in _FIELDS}


def _label(data: Mapping[str, Any]) -> str:
    """Return the entry id for messages, falling back to ``(unnamed)``."""
    value = data.get("id")
    return value if isinstance(value, str) and value else "(unnamed)"


def _text(value: Any) -> bool:
    """Return True when ``value`` is a non-empty string."""
    return isinstance(value, str) and bool(value.strip())


def _check_required(data: Mapping[str, Any]) -> None:
    """Reject an entry that omits or blanks a required text field."""
    missing = [key for key in _REQUIRED if not _text(data.get(key))]
    if missing:
        raise HubCatalogError(
            _label(data), f"missing required fields: {', '.join(missing)}"
        )


def _check_id(data: Mapping[str, Any]) -> None:
    """Reject an id that could escape or confuse the cache layout."""
    entry_id = str(data.get("id", ""))
    if not _ID.match(entry_id):
        raise HubCatalogError(
            _label(data), "id must match [A-Za-z0-9][A-Za-z0-9._/-]*"
        )
    if ".." in entry_id:
        raise HubCatalogError(_label(data), "id must not contain '..'")


def _check_allowed(data: Mapping[str, Any]) -> None:
    """Report an unknown framework, kind, or source by name."""
    for key, allowed in (
        ("framework", FRAMEWORKS),
        ("kind", KINDS),
        ("source", SOURCES),
    ):
        value = data.get(key)
        if value not in allowed:
            raise HubCatalogError(
                _label(data),
                f"unknown {key} {value!r}; expected one of "
                f"{', '.join(allowed)}",
            )


def _check_license(data: Mapping[str, Any]) -> None:
    """Reject a license that is neither a concrete id nor the marker.

    ``"see upstream"`` and other free-text escapes are refused so an entry can
    never present an unverified license as if it were a concrete one. A remote
    entry that is not yet verified must declare the explicit
    :data:`UNVERIFIED_CANDIDATE` marker, which loads but is reported
    unavailable by :func:`spikeforge_hub.catalog.availability`.
    """
    value = data.get("license")
    if value == UNVERIFIED_CANDIDATE:
        return
    if isinstance(value, str) and _LICENSE.match(value):
        return
    raise HubCatalogError(
        _label(data),
        f"license {value!r} is not a concrete SPDX-style id or the "
        f"{UNVERIFIED_CANDIDATE!r} marker",
    )


def _check_source(data: Mapping[str, Any]) -> None:
    """Reject a source whose required locator field is missing."""
    source = data.get("source")
    if source == "url" and not _text(data.get("url")):
        raise HubCatalogError(_label(data), "source 'url' needs a 'url'")
    if source == "hf_repo" and not _text(data.get("hf_repo")):
        raise HubCatalogError(
            _label(data), "source 'hf_repo' needs an 'hf_repo'"
        )
    _check_optional(data)


def _check_optional(data: Mapping[str, Any]) -> None:
    """Reject malformed optional checksum and size fields."""
    size = data.get("size_bytes")
    if size is not None and not isinstance(size, int):
        raise HubCatalogError(_label(data), "size_bytes must be an integer")
    sha = data.get("sha256")
    if sha is not None and not _text(sha):
        raise HubCatalogError(
            _label(data), "sha256 must be a non-empty string or null"
        )


def _fields(data: Mapping[str, Any]) -> Dict[str, Any]:
    """Project ``data`` onto the declared schema fields."""
    return {name: data.get(name) for name in _FIELDS}
