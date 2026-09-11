"""Load, validate, and query the bundled curated model-hub catalog.

The catalog renders fully offline. Every entry is validated on load and any
malformed record is reported through :func:`issues` rather than dropped, so a
bad entry is visible in ``spikeforge-hub list`` output instead of
vanishing. The additive ``available`` flag is computed from
:mod:`spikeforge_hub.probe`, exactly as the dataset catalog gates event
datasets on the ``events`` extra.
"""

import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from spikeforge_hub import cache, probe
from spikeforge_hub.entry import UNVERIFIED_CANDIDATE, HubEntry
from spikeforge_hub.errors import HubCatalogError

#: Bumped whenever the on-disk schema changes incompatibly.
SCHEMA_VERSION = 1
#: The bundled catalog file that ships inside this package.
CATALOG_PATH = Path(__file__).with_name("models.json")

_ENTRIES: List[HubEntry] = []
_ISSUES: List[str] = []
_LOADED = False


def _read(path: Path) -> Mapping[str, object]:
    """Read and parse a catalog file, or raise ``HubCatalogError``."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise HubCatalogError(str(path), f"cannot read: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise HubCatalogError(str(path), f"malformed JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise HubCatalogError(str(path), "catalog root must be a JSON object")
    return payload


def load_catalog(path: Path) -> Tuple[List[HubEntry], List[str]]:
    """Return the valid entries in ``path`` plus every validation issue."""
    payload = _read(path)
    if payload.get("version") != SCHEMA_VERSION:
        version = payload.get("version")
        raise HubCatalogError(str(path), f"unsupported version {version!r}")
    raw = payload.get("entries")
    if not isinstance(raw, list):
        raise HubCatalogError(str(path), "'entries' must be a list")
    entries: List[HubEntry] = []
    issues: List[str] = []
    for index, item in enumerate(raw):
        _collect(item, index, entries, issues)
    return entries, issues


def _collect(
    item: object,
    index: int,
    entries: List[HubEntry],
    issues: List[str],
) -> None:
    """Validate one raw item, recording an issue instead of dropping it."""
    if not isinstance(item, Mapping):
        issues.append(f"entry #{index}: not a JSON object")
        return
    try:
        entries.append(HubEntry.from_dict(item))
    except HubCatalogError as exc:
        issues.append(str(exc))


def _ensure() -> None:
    """Load the bundled catalog once, caching entries and issues."""
    global _LOADED
    if _LOADED:
        return
    found, problems = load_catalog(CATALOG_PATH)
    _ENTRIES[:] = found
    _ISSUES[:] = problems
    _LOADED = True


def entries() -> List[HubEntry]:
    """Return every validated catalog entry."""
    _ensure()
    return list(_ENTRIES)


def issues() -> List[str]:
    """Return the validation problems found while loading the catalog."""
    _ensure()
    return list(_ISSUES)


def get(entry_id: str) -> Optional[HubEntry]:
    """Return the entry with ``entry_id``, or ``None`` when absent."""
    _ensure()
    for entry in _ENTRIES:
        if entry.id == entry_id:
            return entry
    return None


def availability(entry: HubEntry) -> Tuple[bool, Optional[str]]:
    """Return whether ``entry`` can be obtained and, if not, the reason.

    An entry marked :data:`~spikeforge_hub.entry.UNVERIFIED_CANDIDATE` is
    never presented as ready: its upstream repository and license are not
    confirmed, so it is reported unavailable with that reason. A live Hugging
    Face entry additionally needs the ``hub`` extra; bundled and direct-URL
    entries need no extra. An unavailable entry is always named, never
    silently reported as available.
    """
    if entry.license == UNVERIFIED_CANDIDATE:
        return (
            False,
            "unverified candidate: upstream repository and license are not "
            "confirmed; verify the source and replace the license before use",
        )
    if entry.source == "hf_repo" and not probe.available():
        return False, "requires the `hub` extra (huggingface_hub)"
    return True, None


def entry_dict(entry: HubEntry) -> Dict[str, object]:
    """Return a JSON-able entry card with availability and cache state."""
    available, reason = availability(entry)
    return {
        **entry.to_dict(),
        "available": available,
        "reason": reason,
        "cached": cache.cached(entry.id),
    }


def list_entries(
    framework: Optional[str] = None,
    kind: Optional[str] = None,
    available: Optional[bool] = None,
) -> List[Dict[str, object]]:
    """Return entry cards, filtered by framework, kind, and availability."""
    cards = [entry_dict(entry) for entry in entries()]
    return [
        card for card in cards if _matches(card, framework, kind, available)
    ]


def _matches(
    card: Dict[str, object],
    framework: Optional[str],
    kind: Optional[str],
    available: Optional[bool],
) -> bool:
    """Return True when ``card`` passes every supplied filter."""
    if framework is not None and card["framework"] != framework:
        return False
    if kind is not None and card["kind"] != kind:
        return False
    if available is None:
        return True
    return card["available"] is available


def catalog() -> List[Dict[str, object]]:
    """Return every entry card (the list the client and CLI render)."""
    return list_entries()


def search(query: str, limit: int = 20) -> List[Dict[str, object]]:
    """Return entry cards whose id, name, framework, or notes match."""
    needle = query.strip().lower()
    if not needle:
        return []
    hits = [card for card in catalog() if needle in _haystack(card)]
    return hits[:limit] if limit > 0 else hits


def _haystack(card: Dict[str, object]) -> str:
    """Return the lower-cased text fields a catalog search scans."""
    keys = ("id", "name", "framework", "notes")
    return " ".join(str(card.get(key, "")) for key in keys).lower()
