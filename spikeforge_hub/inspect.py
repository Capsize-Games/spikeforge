"""Identify and describe a downloaded hub artifact without committing.

Gate one of the import funnel. An artifact is classified as a ``nir_graph``, a
``state_dict``, or opaque ``framework_weights`` and described by its structure
only. The readers live in :mod:`spikeforge_hub.artifact_readers` and the
report type in :mod:`spikeforge_hub.artifact_report`; this module resolves
an entry to a concrete artifact and dispatches to the right reader. Resolution
depends on the entry's source: a ``bundled`` entry is *rendered* from its
shipped topology preset (structure, freshly-initialised weights), while a
``reference`` entry is *loaded* -- it carries trained weights that ship with
the distribution, checksum-verified against the catalog. An artifact that
cannot be identified raises :class:`HubArtifactError` naming the reason.
"""

import hashlib
import os
from typing import Optional

from spikeforge_hub import cache
from spikeforge_hub.artifact_readers import (
    inspect_dir,
    inspect_file,
    read_state_dict,
)
from spikeforge_hub.artifact_report import (
    FRAMEWORK_WEIGHTS,
    NIR_GRAPH,
    STATE_DICT,
    ArtifactReport,
    normalize_nodes,
)
from spikeforge_hub.entry import WEIGHTS_DIR, HubEntry
from spikeforge_hub.errors import HubArtifactError

__all__ = [
    "FRAMEWORK_WEIGHTS",
    "NIR_GRAPH",
    "STATE_DICT",
    "ArtifactReport",
    "inspect_artifact",
    "normalize_nodes",
    "read_state_dict",
    "reference_path",
    "resolve_path",
]

#: Read in blocks so a checksum never loads a whole checkpoint into memory.
_DIGEST_BLOCK = 1 << 20


def _cached_artifact(entry_id: str) -> Optional[str]:
    """Return the first cached artifact file for ``entry_id``, if any."""
    root = os.path.join(cache.cache_root(), cache.slug(entry_id))
    if not os.path.isdir(root):
        return None
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if os.path.isfile(path):
            return path
    return None


def _materialize(entry: HubEntry) -> str:
    """Render a bundled entry's preset to its NIR graph in the cache."""
    if not entry.topology:
        raise HubArtifactError(entry.id, "bundled entry declares no topology")
    from spikeforge.nir_bridge import save_graph, to_nir
    from spikeforge.topology.registry import build_topology

    try:
        spec, module = build_topology(entry.topology)
        graph = to_nir(spec, module)
    except (ValueError, OSError, RuntimeError) as exc:
        detail = f"cannot render preset: {exc}"
        raise HubArtifactError(entry.id, detail) from exc
    path = os.path.join(cache.entry_dir(entry.id), "artifact.json")
    save_graph(graph, path)
    return path


def _digest(path: str) -> str:
    """Return the SHA-256 of the file at ``path``."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(_DIGEST_BLOCK), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_path(entry: HubEntry) -> str:
    """Return the packaged checkpoint for a trained reference entry.

    Unlike a bundled entry, nothing is rebuilt: the trained weights ship with
    the distribution, so this locates the file and verifies it against the
    checksum the catalog pins. A mismatch is reported rather than loaded --
    the whole point of a trained entry is that its bytes are the artifact.
    """
    if not entry.weights:
        raise HubArtifactError(entry.id, "reference entry names no weights")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), WEIGHTS_DIR, entry.weights
    )
    if not os.path.isfile(path):
        raise HubArtifactError(
            entry.id,
            f"packaged weights {entry.weights!r} are missing from this "
            f"install; reinstall spikeforge-hub",
        )
    if entry.sha256:
        found = _digest(path)
        if found != entry.sha256:
            raise HubArtifactError(
                entry.id,
                f"packaged weights {entry.weights!r} do not match the "
                f"catalog checksum (expected {entry.sha256}, got {found})",
            )
    return path


def resolve_path(entry: HubEntry) -> str:
    """Return the artifact path for ``entry``, materializing what it needs.

    A bundled entry ships as a preset rather than bytes, so it is rendered to
    its NIR graph in the cache the first time it is inspected or imported. A
    reference entry already *is* bytes -- trained weights shipped with the
    distribution -- so it is loaded, never rebuilt.
    """
    if entry.source == "bundled":
        return _materialize(entry)
    if entry.source == "reference":
        return reference_path(entry)
    found = _cached_artifact(entry.id)
    if found is None:
        raise HubArtifactError(entry.id, "no cached artifact; download first")
    return found


def inspect_artifact(path: str) -> ArtifactReport:
    """Return the structural report for the artifact at ``path``."""
    if os.path.isdir(path):
        return inspect_dir(path)
    if not os.path.exists(path):
        raise HubArtifactError(path, "artifact not found")
    return inspect_file(path)
