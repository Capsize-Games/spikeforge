"""Persist and load trained spiking-network checkpoints."""

import os
import re
import time

import torch

from snn_interpreter.config import MODEL_DIR

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def _ensure_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def safe_name(name):
    """Sanitise a user-supplied checkpoint name."""
    cleaned = _SAFE.sub("_", (name or "").strip())
    return cleaned or f"model_{int(time.time())}"


def path_for(name):
    """Return the full checkpoint path for a name (.pt)."""
    base = safe_name(name)
    if not base.endswith(".pt"):
        base += ".pt"
    return os.path.join(MODEL_DIR, base)


def save(name, net, meta):
    """Save model weights plus training metadata to disk."""
    _ensure_dir()
    filepath = path_for(name)
    torch.save({
        "state_dict": net.state_dict(),
        "meta": meta,
        "saved_at": time.time(),
    }, filepath)
    return filepath


def load(name):
    """Load a checkpoint dict by name."""
    return torch.load(path_for(name), map_location="cpu", weights_only=False)


def list_models():
    """Return metadata for every saved checkpoint, newest first."""
    _ensure_dir()
    items = []
    for entry in os.scandir(MODEL_DIR):
        if not entry.name.endswith(".pt"):
            continue
        items.append(_describe(entry))
    return sorted(items, key=lambda m: m["saved_at"], reverse=True)


def _describe(entry):
    """Build a summary dict for one checkpoint file."""
    try:
        ckpt = torch.load(entry.path, map_location="cpu",
                          weights_only=False)
        meta = ckpt.get("meta", {})
        saved_at = ckpt.get("saved_at", entry.stat().st_mtime)
    except Exception:
        meta, saved_at = {}, entry.stat().st_mtime
    return {
        "name": entry.name[: -len(".pt")],
        "meta": meta,
        "input_mode": meta.get("input_mode", "raw"),
        "coding": meta.get("coding", "raw"),
        "saved_at": saved_at,
    }


def delete(name):
    """Remove a checkpoint file if it exists."""
    filepath = path_for(name)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
