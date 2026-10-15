"""Transactional local activation with rollback and worker refs.

A digest proves integrity, never authorization: ``ActivationCache``
activates only a verified bundle, writes the new version before
moving the active pointer, and keeps the previous and
last-known-good versions so an interrupted update leaves the old
version usable. Worker checkpoints capture the model hash and
schema so a wrong model for a session is refused. Weights stay
immutable; each session builds a fresh module from the artifact.
"""

import json
import os
from typing import Any, Dict, Mapping, Optional

from spikeforge.serving.errors import BundleFormatError, BundleIntegrityError
from spikeforge.serving.safetensor_bundle import STBundle
from spikeforge.serving.st_format import ST_SUFFIX

#: Version of the activation-cache layout on disk.
CACHE_VERSION = 1


def _atomic_write(path: str, payload: bytes) -> None:
    """Write ``payload`` to ``path`` atomically with an fsync."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    parent = os.open(os.path.dirname(path) or ".", os.O_DIRECTORY)
    try:
        os.fsync(parent)
    finally:
        os.close(parent)


def _read_pointer(path: str) -> Dict[str, Any]:
    """Return the active pointer doc, or an empty doc when absent."""
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
        return loaded if isinstance(loaded, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


class ActivationCache:
    """A local directory of verified model versions with rollback."""

    def __init__(self, root: str) -> None:
        """Point the cache at ``root``, creating it when needed."""
        self._root = root
        os.makedirs(os.path.join(root, "versions"), exist_ok=True)

    def _version_path(self, model_hash: str) -> str:
        """Return the bundle path stored for ``model_hash``."""
        return os.path.join(self._root, "versions", model_hash,
                            "bundle" + ST_SUFFIX)

    def _pointer_path(self) -> str:
        """Return the path of the active-pointer document."""
        return os.path.join(self._root, "active.json")

    def activate(self, bundle_path: str) -> str:
        """Verify, store, and activate ``bundle_path``; return its hash."""
        bundle = STBundle.load(bundle_path)
        model_hash = bundle.model_hash
        target = self._version_path(model_hash)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(bundle_path, "rb") as handle:
            _atomic_write(target, handle.read())
        pointer = _read_pointer(self._pointer_path())
        previous = pointer.get("active_hash")
        if model_hash != previous:
            good = pointer.get("last_known_good") or previous
            doc = {"version": CACHE_VERSION, "active_hash": model_hash,
                   "previous_hash": previous, "last_known_good": good}
            _atomic_write(self._pointer_path(),
                          json.dumps(doc, indent=2).encode())
        return model_hash

    def active_hash(self) -> Optional[str]:
        """Return the active model hash, or ``None`` when unset."""
        return _read_pointer(self._pointer_path()).get("active_hash")

    def active(self) -> Optional[STBundle]:
        """Load the active verified version, or ``None`` when unset."""
        model_hash = self.active_hash()
        if not model_hash:
            return None
        return STBundle.load(self._version_path(str(model_hash)))

    def rollback(self) -> str:
        """Re-activate the previous version; return its hash."""
        pointer = _read_pointer(self._pointer_path())
        fallback = pointer.get("previous_hash")
        fallback = fallback or pointer.get("last_known_good")
        if not fallback:
            raise BundleFormatError(self._root, "no rollback version")
        doc = {"version": CACHE_VERSION, "active_hash": fallback,
               "previous_hash": pointer.get("active_hash"),
               "last_known_good": fallback}
        _atomic_write(self._pointer_path(), json.dumps(doc, indent=2).encode())
        return str(fallback)


def model_ref(bundle: STBundle) -> Dict[str, Any]:
    """Return the worker-checkpoint model reference for ``bundle``."""
    return {
        "model_hash": bundle.model_hash,
        "schema": "safetensor-bundle-v2",
        "topology": bundle.manifest.get("topology"),
        "encode_spec_version": bundle.manifest.get("encode_spec_version"),
    }


def check_model_ref(ref: Mapping[str, Any], bundle: STBundle) -> None:
    """Raise when ``ref`` does not describe ``bundle``'s artifact."""
    if not isinstance(ref, Mapping):
        raise BundleFormatError("<checkpoint>", "model ref not an object")
    if ref.get("schema") != "safetensor-bundle-v2":
        raise BundleFormatError("<checkpoint>", "model schema mismatch")
    if ref.get("model_hash") != bundle.model_hash:
        raise BundleIntegrityError("<checkpoint>", "weights",
                                   "wrong model hash for session")


def fresh_weights(bundle: STBundle) -> Dict[str, Any]:
    """Return detached weight clones so workers never share tensors."""
    import torch

    return {name: val.detach().clone()
            for name, val in bundle.weights.items()
            if torch.is_tensor(val)}
