"""Diff the metadata of two checkpoints for the records CLI and server.

The diff is the union of both checkpoints' stored ``meta`` mappings, with each
key classified as ``added`` (right only), ``removed`` (left only), ``changed``
(both, differing), or ``same`` (both, equal). When both checkpoints carry a
manifest their config hashes are compared too. The result is JSON-able and the
keys are sorted, so a diff is stable and can be rendered or tested directly.
"""

import os
from typing import Any, Dict, List, Mapping, Optional

from snn_interpreter.network import model_store

#: Key present only on the right side of the diff.
ADDED = "added"
#: Key present only on the left side of the diff.
REMOVED = "removed"
#: Key present on both sides with different values.
CHANGED = "changed"
#: Key present on both sides with equal values.
SAME = "same"

_MISSING = object()


def _classify(left: Any, right: Any) -> str:
    """Return the status label for a metadata key's two values."""
    if left is _MISSING:
        return ADDED
    if right is _MISSING:
        return REMOVED
    return SAME if left == right else CHANGED


def _entry(key: str, left: Any, right: Any) -> Dict[str, Any]:
    """Return one classified key entry for the diff."""
    return {
        "key": key,
        "left": None if left is _MISSING else left,
        "right": None if right is _MISSING else right,
        "status": _classify(left, right),
    }


def _keys(entries: List[Dict[str, Any]], status: str) -> List[str]:
    """Return the keys whose entry carries ``status``."""
    return [item["key"] for item in entries if item["status"] == status]


def _side(name: Optional[str], meta: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the JSON-able header describing one side of the diff."""
    return {"name": name, "meta": dict(meta)}


def metadata_diff(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    left_name: Optional[str] = None,
    right_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the union of both metadata mappings with per-key status.

    Classification is ``added``, ``removed``, ``changed``, or ``same``; the
    parallel ``changed``/``added``/``removed`` lists make a diff easy to
    assert on.
    """
    keys = sorted(set(left) | set(right))
    entries = [
        _entry(key, left.get(key, _MISSING), right.get(key, _MISSING))
        for key in keys
    ]
    return {
        "left": _side(left_name, left),
        "right": _side(right_name, right),
        "entries": entries,
        "changed": _keys(entries, CHANGED),
        "added": _keys(entries, ADDED),
        "removed": _keys(entries, REMOVED),
        "identical": all(item["status"] == SAME for item in entries),
    }


def _checkpoint(name: str) -> Dict[str, Any]:
    """Load a checkpoint or raise ``FileNotFoundError`` naming it."""
    if not os.path.exists(model_store.path_for(name)):
        raise FileNotFoundError(f"no checkpoint named {name!r}")
    return model_store.load(name)


def _hash_pair(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
    """Return both manifests' config hashes with a match flag."""
    left_hash = (left.get("manifest") or {}).get("config_hash")
    right_hash = (right.get("manifest") or {}).get("config_hash")
    return {
        "left": left_hash,
        "right": right_hash,
        "matches": left_hash is not None and left_hash == right_hash,
    }


def checkpoint_diff(left_name: str, right_name: str) -> Dict[str, Any]:
    """Return a metadata diff between two saved checkpoints.

    The diff covers each checkpoint's stored ``meta``; when both carry a
    manifest the config hashes are compared too, so two runs with identical
    settings report ``config_hash.matches`` even if their weights differ. A
    missing checkpoint raises ``FileNotFoundError`` for the caller to report.
    """
    left = _checkpoint(left_name)
    right = _checkpoint(right_name)
    result = metadata_diff(
        left.get("meta") or {},
        right.get("meta") or {},
        left_name=left_name,
        right_name=right_name,
    )
    result["config_hash"] = _hash_pair(left, right)
    return result
