"""Storage of the governed registry document on disk.

Split out of :mod:`spikeforge_hub.registry` so the governance layer stays
within the file-length limit. Loading validates every entry and collects
each problem instead of dropping it, so :meth:`Registry.issues` names
everything wrong with a document while :meth:`Registry.entries` returns
only the valid ones.
"""

import json
import os
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Tuple

from spikeforge.config import REGISTRY_DIR
from spikeforge_hub._validation import label as _label
from spikeforge_hub.registry_entry import RegistryEntry, stage_rank
from spikeforge_hub.registry_errors import (
    RegistryError,
    RegistrySchemaError,
    RegistryVersionError,
)
from spikeforge_hub.registry_schema import (
    REGISTRY_FILE,
    REGISTRY_SCHEMA_VERSION,
)


def default_registry_path() -> str:
    """Return the default registry document path under ``REGISTRY_DIR``."""
    return os.path.join(REGISTRY_DIR, REGISTRY_FILE)


def _read_json_file(path: str) -> Any:
    """Return the parsed JSON at ``path``, wrapping I/O and parse errors."""
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except OSError as exc:
        raise RegistrySchemaError(
            "(registry)", f"cannot read {path}: {exc}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise RegistrySchemaError(
            "(registry)", f"malformed JSON: {exc}"
        ) from exc


def _check_registry_version(data: Mapping[str, Any]) -> int:
    """Return the document's schema version, or raise when unsupported."""
    version = int(data.get("version", REGISTRY_SCHEMA_VERSION))
    if version != REGISTRY_SCHEMA_VERSION:
        raise RegistryVersionError(
            "(registry)",
            f"document version {version!r} is not "
            f"{REGISTRY_SCHEMA_VERSION}",
        )
    return version


def _position(index: int, item: Any, error: RegistryError) -> str:
    """Return a positional prefix for an entry that failed validation."""
    label = _label(item) if isinstance(item, Mapping) else f"#{index}"
    return f"{label}: {error.detail}"


def _parse_registry_entries(
    raw: Any,
) -> Tuple[Tuple[RegistryEntry, ...], Tuple[str, ...]]:
    """Parse each raw entry, collecting a message for every failure."""
    if not isinstance(raw, list):
        raise RegistrySchemaError("(registry)", "'entries' must be a list")
    found: List[RegistryEntry] = []
    problems: List[str] = []
    for index, item in enumerate(raw):
        try:
            found.append(RegistryEntry.from_dict(item))
        except RegistryError as error:
            problems.append(_position(index, item, error))
    return tuple(found), tuple(problems)


@dataclass(frozen=True)
class Registry:
    """A collection of governed entries with deterministic ordering.

    Loading validates every entry and collects each problem instead of
    dropping it, so :meth:`issues` names everything wrong with a document while
    :meth:`entries` returns only the valid ones — the same honest contract the
    catalog uses.
    """

    entries: Tuple[RegistryEntry, ...] = ()
    issues: Tuple[str, ...] = ()
    version: int = REGISTRY_SCHEMA_VERSION

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Registry":
        """Validate a registry document, collecting every entry issue."""
        if not isinstance(data, Mapping):
            raise RegistrySchemaError("(registry)", "root must be a mapping")
        _check_registry_version(data)
        found, problems = _parse_registry_entries(data.get("entries", []))
        return cls(entries=found, issues=problems)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "Registry":
        """Load the registry at ``path`` (or the default path).

        A missing file yields an empty, valid registry so a first promotion
        does not need a pre-seeded document.
        """
        resolved = path or default_registry_path()
        if not os.path.exists(resolved):
            return cls()
        return cls.from_dict(_read_json_file(resolved))

    def save(self, path: Optional[str] = None) -> str:
        """Write the registry to ``path`` and return that path."""
        resolved = path or default_registry_path()
        directory = os.path.dirname(resolved)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(resolved, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(self.to_dict(), indent=2) + "\n")
        return resolved

    def ordered(self) -> Tuple[RegistryEntry, ...]:
        """Return entries ordered deterministically by stage, id, version."""
        return tuple(
            sorted(
                self.entries,
                key=lambda item: (
                    stage_rank(item.stage),
                    item.id,
                    item.version,
                ),
            )
        )

    def get(self, entry_id: str) -> Optional[RegistryEntry]:
        """Return ``entry_id``'s entry at its furthest stage, or None."""
        matches = [item for item in self.entries if item.id == entry_id]
        if not matches:
            return None
        return max(matches, key=lambda item: stage_rank(item.stage))

    def add(self, entry: RegistryEntry) -> "Registry":
        """Return a registry with ``entry`` replacing a same-stage peer."""
        entry.validated()
        kept = tuple(
            item
            for item in self.entries
            if not (item.id == entry.id and item.stage == entry.stage)
        )
        return replace(self, entries=kept + (entry,))

    def to_dict(self) -> Dict[str, Any]:
        """Return the registry as a JSON-able mapping in schema order."""
        return {
            "version": int(self.version),
            "entries": [item.to_dict() for item in self.ordered()],
        }

    def flagged(self) -> List[Dict[str, Any]]:
        """Return the deprecated/retired entries that need attention."""
        return [
            {"id": item.id, "stage": item.stage, "status": item.status}
            for item in self.ordered()
            if item.status != "active"
        ]
