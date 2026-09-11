"""Identify and describe a downloaded hub artifact without committing.

Gate one of the import funnel. An artifact is classified as a ``nir_graph``, a
``state_dict``, or opaque ``framework_weights`` and described by its structure
only. The readers live in :mod:`spikeforge_hub.artifact_readers` and the
report type in :mod:`spikeforge_hub.artifact_report`; this module resolves
an entry to a concrete artifact (materializing a bundled preset) and dispatches
to the right reader. An artifact that cannot be identified raises
:class:`HubArtifactError` naming the reason.
"""

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
from spikeforge_hub.entry import HubEntry
from spikeforge_hub.errors import HubArtifactError

__all__ = [
    "FRAMEWORK_WEIGHTS",
    "NIR_GRAPH",
    "STATE_DICT",
    "ArtifactReport",
    "inspect_artifact",
    "normalize_nodes",
    "read_state_dict",
    "resolve_path",
]


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


def resolve_path(entry: HubEntry) -> str:
    """Return the artifact path for ``entry``, materializing a bundled one.

    A bundled entry ships as a preset rather than bytes, so it is rendered to
    its NIR graph in the cache the first time it is inspected or imported.
    """
    if entry.source == "bundled":
        return _materialize(entry)
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
