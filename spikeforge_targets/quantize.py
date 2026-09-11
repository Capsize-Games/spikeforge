"""Apply a target's declared quantization to a graph's weights.

The ``quantization`` entry in :attr:`TargetSpec.constraints` stays the source
of truth for *which* scheme applies; this module only consumes it. A target
that declares ``none`` yields an unchanged graph and a no-op report, an
unknown scheme is reported unapplied rather than guessed, and a supported
scheme quantizes every weight-bearing node, recording per-layer before/after
ranges and (when a spike fixture is supplied) the measured drift. No device or
backend is involved, so a device-quantized result is never claimed.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.quantize_report import QuantizationReport
from spikeforge_targets.quantize_result import QuantizationResult
from spikeforge_targets.quantize_schemes import SCHEMES, Scheme
from spikeforge_targets.registry import get_target
from spikeforge_targets.rewrite_drift import rewrite_drift
from spikeforge_targets.target_spec import TargetSpec

#: Reason recorded when a target declares no quantization at all.
NO_QUANTIZATION = "target declares no quantization"
#: Reason recorded for a scheme name this module cannot apply.
UNKNOWN_SCHEME = "unknown quantization scheme {scheme!r}"

_Layer = Dict[str, Any]


def _resolve(target: Any) -> TargetSpec:
    """Return the :class:`TargetSpec` for a name or a spec object."""
    if isinstance(target, TargetSpec):
        return target
    return get_target(str(target))


def _as_graph(graph_or_spec: Any) -> Any:
    """Return a NIR graph, exporting a :class:`TopologySpec` when given one."""
    if isinstance(graph_or_spec, TopologySpec):
        return to_nir(graph_or_spec)
    return graph_or_spec


def _weight(node: Any) -> Optional[np.ndarray]:
    """Return a node's floating weight array, or ``None`` when it has none."""
    value = getattr(node, "weight", None)
    if value is None:
        return None
    array = np.asarray(value)
    return array if array.dtype.kind == "f" else None


def _layer(name: str, node: Any, before: Any, after: Any) -> _Layer:
    """Return one per-layer before/after range record."""
    return {
        "node": name,
        "primitive": type(node).__name__,
        "before": [float(before[0]), float(before[1])],
        "after": [float(after[0]), float(after[1])],
    }


def _one(name: str, node: Any, scheme: Scheme) -> Tuple[Any, Optional[_Layer]]:
    """Return ``node`` with quantized weights and its layer record."""
    weight = _weight(node)
    if weight is None:
        return node, None
    quantized, before, after = scheme(weight)
    return replace(node, weight=quantized), _layer(name, node, before, after)


def _nodes(graph: Any, scheme: Scheme) -> Tuple[Dict[str, Any], List[_Layer]]:
    """Return quantized nodes and their per-layer records."""
    nodes: Dict[str, Any] = {}
    layers: List[_Layer] = []
    for name, node in graph.nodes.items():
        new_node, record = _one(name, node, scheme)
        nodes[name] = new_node
        if record is not None:
            layers.append(record)
    return nodes, layers


def _new_graph(nodes: Dict[str, Any], edges: Any) -> Any:
    """Return a fresh, unchecked NIR graph from ``nodes`` and ``edges``."""
    cls = api.node_class("NIRGraph")
    return cls(dict(nodes), list(edges), type_check=False)


def _unapplied(
    graph: Any, spec: TargetSpec, scheme: str, reason: str
) -> QuantizationResult:
    """Return an unchanged graph paired with a named unapplied report."""
    report = QuantizationReport(spec.name, scheme, False, reason)
    return QuantizationResult(graph, report)


def quantize(
    graph_or_spec: Any, target: Any, spikes: Optional[Any] = None
) -> QuantizationResult:
    """Return the quantized graph for ``target`` plus its honest report.

    ``graph_or_spec`` is a NIR graph or a :class:`TopologySpec`; ``target`` a
    name or :class:`TargetSpec`. Supplying ``spikes`` adds the post-quantize
    drift check, executing both graphs with the reference interpreter.
    """
    spec = _resolve(target)
    graph = _as_graph(graph_or_spec)
    scheme_name = str(spec.constraints.get("quantization", "none"))
    if scheme_name == "none":
        return _unapplied(graph, spec, scheme_name, NO_QUANTIZATION)
    scheme = SCHEMES.get(scheme_name)
    if scheme is None:
        reason = UNKNOWN_SCHEME.format(scheme=scheme_name)
        return _unapplied(graph, spec, scheme_name, reason)
    nodes, layers = _nodes(graph, scheme)
    ready = _new_graph(nodes, graph.edges)
    drift = None if spikes is None else rewrite_drift(graph, ready, spikes)
    report = QuantizationReport(
        spec.name, scheme_name, True, "", tuple(layers), drift
    )
    return QuantizationResult(ready, report)
