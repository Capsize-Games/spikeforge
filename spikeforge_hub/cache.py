"""Offline cache locations and size accounting for hub artifacts.

The cache lives under :data:`spikeforge.config.HUB_CACHE_DIR` (env
``SPIKEFORGE_HUB_DIR``), alongside ``MODEL_DIR`` and outside the
trained-model store, so an imported foreign artifact never lands in the
trained-model registry
until it is explicitly promoted.
"""

import os

from spikeforge.config import HUB_CACHE_DIR


def cache_root() -> str:
    """Return the configured hub cache root directory."""
    return HUB_CACHE_DIR


def ensure_dir(path: str) -> str:
    """Create ``path`` and its parents, returning ``path``."""
    os.makedirs(path, exist_ok=True)
    return path


def slug(entry_id: str) -> str:
    """Map a catalog id such as ``nir/conv_net`` to a directory name."""
    return entry_id.replace("/", "__").strip()


def entry_dir(entry_id: str) -> str:
    """Return (and create) the cache directory for a catalog entry."""
    return ensure_dir(os.path.join(cache_root(), slug(entry_id)))


def dir_bytes(path: str) -> int:
    """Sum the sizes of every file under ``path`` (best effort)."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def entry_bytes(entry_id: str) -> int:
    """Return the bytes currently cached for ``entry_id``."""
    path = os.path.join(cache_root(), slug(entry_id))
    return dir_bytes(path) if os.path.isdir(path) else 0


def cached(entry_id: str) -> bool:
    """Return True when any artifact bytes are cached for ``entry_id``."""
    return entry_bytes(entry_id) > 0
