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
    BUNDLE_SUFFIXES,
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
    """Return the tensor mapping stored in the artifact at ``path``."""
    if os.path.splitext(path)[1].lower() in BUNDLE_SUFFIXES:
        return read_bundle_state_dict(path)
    state = _state_dict(_load_torch(path))
    if state is None:
        raise HubArtifactError(path, "torch object carries no state dict")
    return dict(state)


def _load_bundle(path: str) -> Any:
    """Load and integrity-check the ``.spkf`` bundle at ``path``."""
    from spikeforge.serving.bundle import DeploymentBundle

    try:
        return DeploymentBundle.load(path)
    except Exception as exc:  # integrity errors must not escape untagged
        raise HubArtifactError(path, f"unreadable bundle: {exc}") from exc


def read_bundle_state_dict(path: str) -> Dict[str, Any]:
    """Return the float tensor mapping stored in the bundle at ``path``."""
    try:
        return dict(_load_bundle(path).resolved_weights())
    except HubArtifactError:
        raise
    except Exception as exc:  # undecodable payloads must not escape untagged
        raise HubArtifactError(
            path, f"bundle weights do not decode: {exc}"
        ) from exc


def _bundle_classes(manifest: Any) -> Optional[int]:
    """Return the class count the bundle manifest declares, if any.

    Only the manifest card is consulted, never the tensor shapes, for the
    same reason :func:`_declared_classes` never reads shapes back: a count
    derived from the weights could never disagree with them.
    """
    if not isinstance(manifest, Mapping):
        return None
    declared = manifest.get("num_classes")
    if isinstance(declared, bool) or not isinstance(declared, int):
        return None
    return declared if declared > 0 else None


def inspect_bundle(path: str) -> ArtifactReport:
    """Describe a ``.spkf`` bundle by its manifest and weight shapes."""
    bundle = _load_bundle(path)
    try:
        weights = bundle.resolved_weights()
    except Exception as exc:  # undecodable payloads must not escape untagged
        raise HubArtifactError(
            path, f"bundle weights do not decode: {exc}"
        ) from exc
    note = "spkf bundle, integrity-checked against its manifest"
    return ArtifactReport(
        STATE_DICT, path, (), _key_records(weights), (), (note,),
        _bundle_classes(bundle.manifest),
    )


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


def _declared_classes(loaded: Any) -> Optional[int]:
    """Return the class count a checkpoint's metadata card declares.

    Only the card is consulted, never the tensor shapes: a count read back
    from the shapes would agree with them by construction and so could never
    disagree, which is the opposite of what a compatibility check is for.
    A bare state dict has no card and returns ``None``.
    """
    if not isinstance(loaded, Mapping):
        return None
    meta = loaded.get("meta")
    if not isinstance(meta, Mapping):
        return None
    declared = meta.get("num_classes")
    if isinstance(declared, bool) or not isinstance(declared, int):
        return None
    return declared if declared > 0 else None


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
        STATE_DICT, path, (), _key_records(state), (), ("torch state dict",),
        _declared_classes(loaded),
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
    if suffix in BUNDLE_SUFFIXES:
        return inspect_bundle(path)
    raise HubArtifactError(path, f"unrecognized artifact extension {suffix!r}")
