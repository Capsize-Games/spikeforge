"""Persist and load trained spiking-network checkpoints."""

import os
import re
import time
from typing import Any, Dict, List, Optional

import torch

from spikeforge.config import MODEL_DIR

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


class CheckpointError(Exception):
    """Raised when a checkpoint file exists but is refused a load.

    The checkpoint directory is shared with hub imports and, on a
    default deploy, with bytes that arrived over the socket, so every
    load here is fail-closed: a file that is not plain tensors and
    containers raises this instead of unpickling. The offending
    ``name`` and a human ``detail`` are stored as attributes.
    """

    def __init__(self, name: str, detail: str) -> None:
        """Record ``name`` and ``detail`` and build a clear message."""
        super().__init__(f"checkpoint {name!r} is not loadable: {detail}")
        self.name: str = name
        self.detail: str = detail


def _ensure_dir() -> None:
    """Create the model directory if it does not already exist."""
    os.makedirs(MODEL_DIR, exist_ok=True)


def safe_name(name: Optional[str]) -> str:
    """Sanitise a user-supplied checkpoint name."""
    cleaned = _SAFE.sub("_", (name or "").strip())
    return cleaned or f"model_{int(time.time())}"


def path_for(name: Optional[str]) -> str:
    """Return the full checkpoint path for a name (.pt)."""
    base = safe_name(name)
    if not base.endswith(".pt"):
        base += ".pt"
    return os.path.join(MODEL_DIR, base)


def save(
    name: Optional[str],
    net: torch.nn.Module,
    meta: Dict[str, Any],
    history: Optional[List[Dict[str, Any]]] = None,
    manifest: Optional[Dict[str, Any]] = None,
) -> str:
    """Save weights, metadata, metric history, and an optional manifest."""
    _ensure_dir()
    filepath = path_for(name)
    torch.save(
        {
            "state_dict": net.state_dict(),
            "meta": meta,
            "history": history or [],
            "manifest": manifest,
            "saved_at": time.time(),
        },
        filepath,
    )
    return filepath


def load(name: Optional[str]) -> Dict[str, Any]:
    """Load a checkpoint dict by name.

    Raises a plain ``FileNotFoundError`` with the checkpoint name (not the
    raw disk path) so a client that surfaces ``str(exc)`` verbatim shows
    something a user can act on instead of a server filesystem path.
    A file that is not plain tensors and containers raises
    :class:`CheckpointError` instead of unpickling: checkpoint bytes
    are untrusted, so the load is fail-closed.
    """
    filepath = path_for(name)
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"checkpoint {name!r} not found (it may have been deleted)"
        )
    try:
        return torch.load(filepath, map_location="cpu", weights_only=True)
    except Exception as exc:  # refused pickles must not escape untagged
        raise CheckpointError(
            name or "", f"refused unsafe load: {exc}"
        ) from exc


def load_checkpoint_meta(name: Optional[str]) -> Dict[str, Any]:
    """Return a checkpoint's stored ``meta`` mapping, or ``{}`` if missing.

    Only a missing checkpoint file falls back to ``{}``. A file that exists
    but refuses to load raises :class:`CheckpointError` naming it, so a
    corrupt checkpoint can never be mistaken for one without metadata.
    """
    if not name:
        return {}
    try:
        return load(name).get("meta", {})
    except FileNotFoundError:
        return {}


def manifest(name: Optional[str]) -> Dict[str, Any]:
    """Return a checkpoint's stored reproducibility manifest.

    The result always carries an ``available`` flag. A legacy checkpoint saved
    before manifests existed, or a missing file, yields a safe fallback whose
    ``reason`` names the cause and whose ``meta`` echoes what is on disk, so a
    reader never has to treat absence as an error.
    """
    try:
        ckpt = load(name)
    except Exception as exc:
        return {"available": False, "reason": f"checkpoint unavailable: {exc}"}
    stored = ckpt.get("manifest")
    if stored:
        return {**stored, "available": True}
    return {
        "available": False,
        "reason": "no manifest stored (legacy checkpoint)",
        "meta": ckpt.get("meta", {}),
    }


def list_models() -> List[Dict[str, Any]]:
    """Return metadata for every saved checkpoint, newest first."""
    _ensure_dir()
    items: List[Dict[str, Any]] = []
    for entry in os.scandir(MODEL_DIR):
        if not entry.name.endswith(".pt"):
            continue
        items.append(_describe(entry))
    return sorted(items, key=lambda item: item["saved_at"], reverse=True)


def _describe(entry: os.DirEntry) -> Dict[str, Any]:
    """Build a summary dict for one checkpoint file.

    The load is fail-closed: a file that is not plain tensors and
    containers is skipped and reported with ``unreadable`` and
    ``error`` instead of unpickling, so one planted file can neither
    execute code on a read-only listing nor hide silently.
    """
    extra: Dict[str, Any] = {}
    try:
        ckpt = torch.load(entry.path, map_location="cpu", weights_only=True)
        meta = ckpt.get("meta", {})
        saved_at = ckpt.get("saved_at", entry.stat().st_mtime)
    except Exception as exc:
        meta, saved_at = {}, entry.stat().st_mtime
        extra = {
            "unreadable": True,
            "error": f"refused unsafe load: {exc}",
        }
    return {
        "name": entry.name[: -len(".pt")],
        "meta": meta,
        "input_mode": meta.get("input_mode", "raw"),
        "coding": meta.get("coding", "raw"),
        "topology": meta.get("topology", "fc_legacy"),
        "saved_at": saved_at,
        **extra,
    }


def delete(name: Optional[str]) -> bool:
    """Remove a checkpoint file if it exists."""
    filepath = path_for(name)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
