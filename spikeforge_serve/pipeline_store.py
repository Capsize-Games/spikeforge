"""Persist and load saved pipeline graphs.

Mirrors :mod:`spikeforge.network.model_store`'s shape exactly, but for a
pipeline's small JSON document rather than a ``.pt`` checkpoint.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from spikeforge.config import DATA_DIR

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")

#: Override with SPIKEFORGE_PIPELINES_DIR to relocate saved pipelines.
PIPELINES_DIR = (
    os.environ.get("SPIKEFORGE_PIPELINES_DIR")
    or os.path.join(DATA_DIR, "pipelines")
)


def _ensure_dir() -> None:
    """Create the pipelines directory if it does not already exist."""
    os.makedirs(PIPELINES_DIR, exist_ok=True)


def safe_name(name: Optional[str]) -> str:
    """Sanitise a user-supplied pipeline name."""
    cleaned = _SAFE.sub("_", (name or "").strip())
    return cleaned or f"pipeline_{int(time.time())}"


def path_for(name: Optional[str]) -> str:
    """Return the full pipeline file path for a name (.json)."""
    base = safe_name(name)
    if not base.endswith(".json"):
        base += ".json"
    return os.path.join(PIPELINES_DIR, base)


def save(name: str, graph: Dict[str, Any]) -> str:
    """Persist a pipeline graph dict; return the path it was written to."""
    _ensure_dir()
    filepath = path_for(name)
    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(graph, handle, indent=2)
    return filepath


def load(name: str) -> Dict[str, Any]:
    """Load a saved pipeline graph by name.

    Raises ``FileNotFoundError`` with the pipeline name (not the raw disk
    path) so a client surfacing the message verbatim shows something
    actionable.
    """
    filepath = path_for(name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"pipeline {name!r} not found (it may have been deleted)"
        )
    with open(filepath, encoding="utf-8") as handle:
        return json.load(handle)


def list_pipelines() -> List[Dict[str, Any]]:
    """Return summary metadata for every saved pipeline, newest first."""
    _ensure_dir()
    items: List[Dict[str, Any]] = []
    for entry in os.scandir(PIPELINES_DIR):
        if not entry.name.endswith(".json"):
            continue
        try:
            with open(entry.path, encoding="utf-8") as handle:
                graph = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        items.append({
            "name": entry.name[: -len(".json")],
            "node_count": len(graph.get("nodes") or []),
            "edge_count": len(graph.get("edges") or []),
            "saved_at": entry.stat().st_mtime,
        })
    return sorted(items, key=lambda item: item["saved_at"], reverse=True)


def delete(name: str) -> bool:
    """Remove a saved pipeline file if it exists."""
    filepath = path_for(name)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
