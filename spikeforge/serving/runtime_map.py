"""Map a verified bundle onto the supported native runtime shape.

The C runtime parses a ``graph_summary``-shaped JSON value with
inline weight arrays, not the Python NIR envelope: this module
renders that shape from the verified spec and weights, enumerates
the actually emitted nodes for capability checks, and describes
the versioned binary-tensor-reference extension as a capability.
Tensor sources are never silently duplicated: a preserved envelope
must agree on node names or the mapping is refused.
"""

from typing import Any, Dict, List, Mapping

from spikeforge.serving.errors import BundleFormatError
from spikeforge.serving.safetensor_bundle import STBundle

#: Node kinds the native runtime implements (see graph.h).
RUNTIME_KINDS = frozenset({
    "Input", "Output", "Conv2d", "Affine", "Flatten", "AvgPool2d",
    "SumPool2d", "LI", "Threshold", "Scale", "Delay",
})

#: Versioned descriptor of the binary tensor-reader extension.
BINARY_CAPABILITY = {
    "capability": "safetensors-ref",
    "version": 1,
    "status": "proposed",
    "weights_entry": "weights.safetensors",
    "format": "spikeforge-safetensor-bundle",
    "bundle_version": 2,
}


def describe_binary_capability() -> Dict[str, Any]:
    """Return the versioned binary-reader capability descriptor."""
    return dict(BINARY_CAPABILITY)


def map_to_runtime_summary(bundle: STBundle) -> Dict[str, Any]:
    """Render the runtime's summary shape from verified weights."""
    from spikeforge.nir_bridge.exporter import graph_summary, to_nir
    from spikeforge.serving.st_build import module_for

    _, module = module_for(dict(bundle.model["spec"]), bundle.weights)
    summary = graph_summary(to_nir(bundle.spec, module))
    _check_kinds(bundle.path or "<memory>", summary)
    _check_envelope(bundle, summary)
    return summary


def _check_kinds(path: str, summary: Mapping[str, Any]) -> None:
    """Raise naming the first node kind the runtime cannot run."""
    for node in summary.get("nodes") or []:
        kind = node.get("kind")
        if kind not in RUNTIME_KINDS:
            raise BundleFormatError(path, f"unsupported node: {kind}")


def _check_envelope(bundle: STBundle, summary: Mapping[str, Any]) -> None:
    """Raise when the preserved envelope disagrees on node names."""
    graph = bundle.graph
    if not graph:
        return
    try:
        kept = set((graph.get("graph") or {}).get("nodes") or {})
    except AttributeError:
        raise BundleFormatError(bundle.path or "<m>", "bad envelope") from None
    fresh = {node.get("name") for node in summary.get("nodes") or []}
    if kept and kept != fresh:
        raise BundleFormatError(bundle.path or "<m>", "graph drift")


def capability_report(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Report the emitted nodes against the runtime's supported set."""
    kinds = sorted({n.get("kind") for n in summary.get("nodes") or []})
    return {
        "node_count": len(summary.get("nodes") or []),
        "edge_count": len(summary.get("edges") or []),
        "node_kinds": kinds,
        "unsupported": sorted(set(kinds) - RUNTIME_KINDS),
        "binary_capability": describe_binary_capability(),
    }


def emitted_nodes(summary: Mapping[str, Any]) -> List[str]:
    """Return the emitted node names in summary order."""
    return [str(n.get("name")) for n in summary.get("nodes") or []]
