"""Read one artifact file into a structural :class:`ArtifactReport`.

Every reader is honest by construction: a file it cannot read raises
:class:`HubArtifactError` naming the reason, and a format it cannot decode
structurally (opaque framework weights, a directory snapshot) is reported as
:data:`FRAMEWORK_WEIGHTS` rather than guessed at.
"""

import os
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge_hub.artifact_report import (
    FRAMEWORK_WEIGHTS,
    NIR_GRAPH,
    NIR_SUFFIXES,
    STATE_DICT,
    TORCH_SUFFIXES,
    WEIGHT_SUFFIXES,
    ArtifactReport,
    normalize_nodes,
)
from spikeforge_hub.errors import HubArtifactError


def _load_torch(path: str) -> Any:
    """Load a torch artifact at ``path`` or raise ``HubArtifactError``."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # pickle/format errors must not escape untagged
        raise HubArtifactError(path, f"unreadable torch file: {exc}") from exc


def _state_dict(loaded: Any) -> Optional[Mapping[str, Any]]:
    """Return the tensor mapping inside a loaded torch object, if any."""
    if not isinstance(loaded, Mapping):
        return None
    inner = loaded.get("state_dict") if "state_dict" in loaded else None
    candidate = inner if isinstance(inner, Mapping) else loaded
    if candidate and all(torch.is_tensor(item) for item in candidate.values()):
        return candidate
    return None


def _mapping_keys(loaded: Any) -> Tuple[Dict[str, Any], ...]:
    """Return a mapping's top-level keys as shape-less key records."""
    if not isinstance(loaded, Mapping):
        return ()
    return tuple({"key": str(key), "shape": []} for key in loaded)


def _key_records(state: Mapping[str, Any]) -> Tuple[Dict[str, Any], ...]:
    """Return ``{key, shape}`` records for a state-dict mapping."""
    return tuple(
        {"key": str(key), "shape": [int(dim) for dim in tensor.shape]}
        for key, tensor in state.items()
    )


def read_state_dict(path: str) -> Dict[str, Any]:
    """Return the tensor mapping stored in the torch artifact at ``path``."""
    state = _state_dict(_load_torch(path))
    if state is None:
        raise HubArtifactError(path, "torch object carries no state dict")
    return dict(state)


def _read_graph(path: str) -> Any:
    """Load the NIR graph at ``path`` or raise ``HubArtifactError``."""
    from spikeforge.nir_bridge import load_graph

    try:
        return load_graph(path)
    except Exception as exc:  # typed graph errors become one hub error
        raise HubArtifactError(path, f"unreadable NIR graph: {exc}") from exc


def inspect_nir(path: str) -> ArtifactReport:
    """Describe a persisted NIR graph by its nodes and edges."""
    from spikeforge.nir_bridge import graph_summary

    summary = graph_summary(_read_graph(path))
    nodes = normalize_nodes(summary["nodes"])
    edges = tuple(
        {
            "source": str(edge["source"]),
            "target": str(edge["target"]),
            "delayed": bool(edge["delayed"]),
        }
        for edge in summary["edges"]
    )
    note = f"NIR graph with {len(nodes)} nodes and {len(edges)} edges"
    return ArtifactReport(NIR_GRAPH, path, nodes, (), edges, (note,))


def inspect_torch(path: str) -> ArtifactReport:
    """Describe a torch checkpoint or bare state dict by key and shape."""
    loaded = _load_torch(path)
    state = _state_dict(loaded)
    if state is None:
        return ArtifactReport(
            FRAMEWORK_WEIGHTS,
            path,
            (),
            _mapping_keys(loaded),
            (),
            ("torch object is not a state dict",),
        )
    return ArtifactReport(
        STATE_DICT, path, (), _key_records(state), (), ("torch state dict",)
    )


def inspect_weights(path: str) -> ArtifactReport:
    """Describe an opaque framework weight file by name and size."""
    try:
        size = int(os.path.getsize(path))
    except OSError as exc:
        raise HubArtifactError(path, f"unreadable weight file: {exc}") from exc
    keys = ({"key": os.path.basename(path), "shape": [size]},)
    note = "opaque framework weights; no structural mapping available"
    return ArtifactReport(FRAMEWORK_WEIGHTS, path, (), keys, (), (note,))


def _dir_files(path: str) -> Tuple[Tuple[str, int], ...]:
    """Return the ``(relative name, size)`` of every file under ``path``."""
    found = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            full = os.path.join(root, name)
            found.append((os.path.relpath(full, path), os.path.getsize(full)))
    return tuple(found)


def inspect_dir(path: str) -> ArtifactReport:
    """Describe a directory snapshot by listing its files by name and size."""
    files = _dir_files(path)
    if not files:
        raise HubArtifactError(path, "directory snapshot is empty")
    keys = tuple({"key": name, "shape": [int(size)]} for name, size in files)
    note = "directory snapshot; per-file structure is not decoded"
    return ArtifactReport(FRAMEWORK_WEIGHTS, path, (), keys, (), (note,))


def inspect_file(path: str) -> ArtifactReport:
    """Dispatch one artifact file to its kind-specific inspector."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in NIR_SUFFIXES:
        return inspect_nir(path)
    if suffix in TORCH_SUFFIXES:
        return inspect_torch(path)
    if suffix in WEIGHT_SUFFIXES:
        return inspect_weights(path)
    raise HubArtifactError(path, f"unrecognized artifact extension {suffix!r}")
