"""JSON-able payloads for the model-hub WebSocket actions.

Mirrors :mod:`server.target_payloads`: the socket surface reuses the same
:mod:`spikeforge_hub` functions the ``spikeforge-hub`` CLI calls, so the two
can never disagree. The catalog listing always works offline; only live
Hugging
Face search is gated on the ``hub`` extra and reports ``available: false`` with
a named reason rather than erroring.
"""

import os
from typing import Any, Dict, Optional

from spikeforge.config import DATA_DIR
from spikeforge.network import model_store
from spikeforge.serving import bundle as bundle_mod
from spikeforge_hub import probe
from spikeforge_hub.catalog import get, issues, list_entries, search
from spikeforge_hub.errors import HubArtifactError
from spikeforge_hub.import_model import import_model
from spikeforge_hub.inspect import inspect_artifact, resolve_path
from spikeforge_hub.verify import file_sha256

#: Reason reported when live Hugging Face search is unavailable.
_EXTRA_REASON = "requires the `hub` extra (huggingface_hub)"
#: Subdirectory of the data root holding dashboard-built ``.spkf`` bundles.
BUNDLES_DIR_NAME = "bundles"


def list_payload(
    framework: Optional[str] = None,
    kind: Optional[str] = None,
    available: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return the filtered catalog plus any catalog validation issues."""
    return {
        "entries": list_entries(framework, kind, available),
        "issues": issues(),
    }


def search_payload(query: str, limit: int = 20) -> Dict[str, Any]:
    """Return catalog search results and live-search availability."""
    live = probe.available()
    return {
        "query": query,
        "limit": limit,
        "available": live,
        "reason": None if live else _EXTRA_REASON,
        "results": search(query, limit),
    }


def inspect_payload(entry_id: str) -> Dict[str, Any]:
    """Return the structural report for ``entry_id``'s resolved artifact."""
    entry = get(entry_id)
    if entry is None:
        raise HubArtifactError(entry_id, "unknown catalog entry")
    return inspect_artifact(resolve_path(entry)).to_dict()


def import_payload(
    entry_id: str, topology: Optional[str] = None
) -> Dict[str, Any]:
    """Return the full three-gate import report for ``entry_id``."""
    return import_model(entry_id=entry_id, topology=topology)


def bundles_dir() -> str:
    """Return (and create) the directory for dashboard-built bundles."""
    path = os.path.join(DATA_DIR, BUNDLES_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def bundle_payload(name: str) -> Dict[str, Any]:
    """Build saved model ``name`` into a ``.spkf`` and report its identity.

    The archive is written to a stable path and rebuilt on every call, so
    the reported ``path``/``size_bytes``/``sha256`` always describe fresh
    bytes the dashboard can then upload to the hub API.
    """
    saved = {entry["name"] for entry in model_store.list_models()}
    if name not in saved:
        raise HubArtifactError(name, f"no saved model named {name!r}")
    out = os.path.join(bundles_dir(), f"{model_store.safe_name(name)}.spkf")
    bundle_mod.build(name, out=out)
    return {
        "name": name,
        "path": out,
        "size_bytes": os.path.getsize(out),
        "sha256": file_sha256(out),
    }
