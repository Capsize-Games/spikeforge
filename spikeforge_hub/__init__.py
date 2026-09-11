"""Public API for the curated model hub.

The offline catalog, capability probe, cache, isolated download worker, and the
inspect/import funnel are wired here. Live Hugging Face access and artifact
import build on this core in later phases; anything unavailable is reported,
never faked.
"""

from typing import Any, Dict, List, Optional

from spikeforge_hub import cache, probe
from spikeforge_hub.catalog import (
    entries,
    get,
    issues,
    list_entries,
    search,
)
from spikeforge_hub.download_cli import download_entry
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import (
    HubArtifactError,
    HubCatalogError,
    HubDownloadCancelledError,
    HubDownloadError,
    HubError,
    HubExtraMissingError,
    HubImportError,
)
from spikeforge_hub.import_model import import_model as _run_import
from spikeforge_hub.inspect import inspect_artifact, resolve_path

__all__ = [
    "HubArtifactError",
    "HubCatalogError",
    "HubDownloadCancelledError",
    "HubDownloadError",
    "HubEntry",
    "HubError",
    "HubExtraMissingError",
    "HubImportError",
    "available",
    "catalog",
    "download",
    "entries",
    "get",
    "import_model",
    "inspect",
    "issues",
    "search",
]


def available() -> bool:
    """Return True when live Hugging Face access (the ``hub`` extra) works."""
    return probe.available()


def catalog() -> List[Dict[str, object]]:
    """Return every entry card, with availability and cache state."""
    return list_entries()


def download(entry_id: str, dest: Optional[str] = None) -> Dict[str, Any]:
    """Download ``entry_id`` synchronously and return the worker report.

    The async :class:`~spikeforge_hub.downloads.HubDownloadManager` runs
    the same worker in a child process with progress and cancellation.
    """
    entry = get(entry_id)
    if entry is None:
        raise HubDownloadError(entry_id, "unknown catalog entry")
    return download_entry(entry, dest or cache.entry_dir(entry.id))


def inspect(entry_id: str) -> Dict[str, Any]:
    """Return the structural report for ``entry_id``'s resolved artifact."""
    entry = get(entry_id)
    if entry is None:
        raise HubArtifactError(entry_id, "unknown catalog entry")
    return inspect_artifact(resolve_path(entry)).to_dict()


def import_model(
    entry_id: str,
    topology: Optional[str] = None,
    promote: bool = True,
) -> Dict[str, Any]:
    """Inspect, classify, and (when compatible) promote a catalog artifact."""
    return _run_import(entry_id=entry_id, topology=topology, promote=promote)
