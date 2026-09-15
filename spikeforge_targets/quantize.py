"""Apply a target's declared quantization to a graph and check its drift.

The ``quantization`` and ``activation_quantization`` entries in
:attr:`TargetSpec.constraints` stay the source of truth for *which* schemes
apply; this module only consumes them. A weight scheme restricts every
weight-bearing node in a fresh graph, recording per-layer before/after
ranges. An activation scheme cannot live in a NIR graph, so it is simulated
in the drift check: the quantized graph is executed by the reference
interpreter under a
:class:`~spikeforge_targets.activation_quant_graph.GraphActivationQuantizer`
hook whose grid is calibrated on the same spike fixture unless a calibration
is supplied, and the drift names what it includes. A target that declares
``none`` yields a no-op report, an unknown scheme is reported unapplied
rather than guessed, and no device or backend is involved, so a
device-quantized result is never claimed.
"""

from dataclasses import replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.require import require_node
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.activation_quant import Calibration, refused_report
from spikeforge_targets.activation_quant_graph import (
    GraphActivationQuantizer,
    calibrate_graph,
)
from spikeforge_targets.quantize_report import Layers, QuantizationReport
from spikeforge_targets.quantize_result import QuantizationResult
from spikeforge_targets.quantize_schemes import SCHEMES, Scheme
from spikeforge_targets.registry import get_target
from spikeforge_targets.rewrite_drift import rewrite_drift
from spikeforge_targets.target_spec import TargetSpec

#: Reason recorded when a target declares no weight quantization at all.
NO_QUANTIZATION = "target declares no quantization"
#: Reason recorded for a weight scheme name this module cannot apply.
UNKNOWN_SCHEME = "unknown quantization scheme {scheme!r}"
#: Reason recorded for an activation scheme that had no fixture to run on.
NO_FIXTURE = (
    "no spike fixture supplied; activation quantization is simulated only "
    "in the drift check"
)
#: Constraint keys naming a target's weight and activation schemes.
WEIGHT_KEY = "quantization"
ACTIVATION_KEY = "activation_quantization"
#: Calibration source recorded when the grid is folded from the fixture.
FIXTURE_SOURCE = "drift fixture"
#: Drift entry naming the weight rounding.
WEIGHTS = "weights"

_Layer = Dict[str, Any]
#: The weight pass: the graph to run, its layer records, applied, reason.
_Weights = Tuple[Any, Layers, bool, str]
#: The weight pass plus the drift block and the activation hook it fed.
_Passes = Tuple[
    Any, Layers, bool, str, Optional[Dict[str, Any]], GraphActivationQuantizer
]


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


def _declared(spec: TargetSpec, key: str) -> str:
    """Return the scheme ``spec`` declares under ``key``, ``none`` absent."""
    return str(spec.constraints.get(key, "none"))


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
    cls = require_node("NIRGraph", "graph")
    return cls(dict(nodes), list(edges), type_check=False)


def _weights(graph: Any, scheme_name: str) -> _Weights:
    """Return the weight-quantized graph, or ``graph`` itself with a reason."""
    if scheme_name == "none":
        return graph, (), False, NO_QUANTIZATION
    scheme = SCHEMES.get(scheme_name)
    if scheme is None:
        return graph, (), False, UNKNOWN_SCHEME.format(scheme=scheme_name)
    nodes, layers = _nodes(graph, scheme)
    return _new_graph(nodes, graph.edges), tuple(layers), True, ""


def _hook(
    requested: str, calibration: Optional[Calibration],
    ready: Any, spikes: Optional[Any],
) -> GraphActivationQuantizer:
    """Return the activation hook, calibrated on ``spikes`` by default.

    Folded from a plain run of the weight-quantized graph, so the grid is
    fixed and never clips; a supplied calibration is used as given.
    """
    hook = GraphActivationQuantizer(requested, calibration=calibration)
    scheme = hook.scheme
    if scheme is None or not hook.applied:
        return hook
    if calibration is not None or spikes is None:
        return hook
    folded = calibrate_graph(
        ready, spikes, bits=scheme.bits, target=scheme.target,
        source=FIXTURE_SOURCE,
    )
    return GraphActivationQuantizer(requested, calibration=folded)


def _drift(
    graph: Any,
    ready: Any,
    spikes: Optional[Any],
    applied: bool,
    hook: GraphActivationQuantizer,
) -> Optional[Dict[str, Any]]:
    """Return the drift of ``ready`` under ``hook`` and what it includes."""
    if spikes is None or not (applied or hook.applied):
        return None
    post_node = hook if hook.applied else None
    drift = rewrite_drift(graph, ready, spikes, post_node=post_node)
    includes = ([WEIGHTS] if applied else []) + list(hook.targets)
    return {**drift, "includes": includes}


def _activation_section(
    hook: GraphActivationQuantizer, requested: str, refusal: Optional[str]
) -> Dict[str, Any]:
    """Return the activation block; an applicable scheme may be refused."""
    if hook.applied and refusal is not None:
        return refused_report(requested, refusal).to_dict()
    return hook.report().to_dict()


def declared_report(target: Any, reason: str) -> QuantizationReport:
    """Return ``target``'s declared schemes reported unapplied for ``reason``.

    Used where there is nothing to quantize (a spec without a built module),
    so the declared schemes are named without claiming they were applied.
    """
    spec = _resolve(target)
    requested = _declared(spec, ACTIVATION_KEY)
    hook = GraphActivationQuantizer(requested)
    return QuantizationReport(
        spec.name,
        _declared(spec, WEIGHT_KEY),
        False,
        reason,
        (),
        None,
        _activation_section(hook, requested, reason),
    )


def _resolve_inputs(
    graph_or_spec: Any, target: Any, activation: Optional[str]
) -> Tuple[TargetSpec, Any, str, str]:
    """Resolve the target spec, graph, and weight/activation scheme names."""
    spec = _resolve(target)
    graph = _as_graph(graph_or_spec)
    weight_name = _declared(spec, WEIGHT_KEY)
    requested = (
        _declared(spec, ACTIVATION_KEY) if activation is None
        else str(activation)
    )
    return spec, graph, weight_name, requested


def _run_passes(
    graph: Any,
    weight_name: str,
    requested: str,
    calibration: Optional[Calibration],
    spikes: Optional[Any],
) -> _Passes:
    """Run the weight and activation passes the report is built from."""
    ready, layers, applied, reason = _weights(graph, weight_name)
    hook = _hook(requested, calibration, ready, spikes)
    drift = _drift(graph, ready, spikes, applied, hook)
    return ready, layers, applied, reason, drift, hook


def _report(
    spec: TargetSpec, weight_name: str, applied: bool, reason: str,
    layers: Layers, drift: Optional[Dict[str, Any]],
    hook: GraphActivationQuantizer, requested: str, spikes: Optional[Any],
) -> QuantizationReport:
    """Assemble the quantization report from the weight/activation passes."""
    refusal = NO_FIXTURE if spikes is None else None
    return QuantizationReport(
        spec.name,
        weight_name,
        applied,
        reason,
        layers,
        drift,
        _activation_section(hook, requested, refusal),
    )


def quantize(
    graph_or_spec: Any, target: Any, spikes: Optional[Any] = None,
    activation: Optional[str] = None,
    calibration: Optional[Calibration] = None,
) -> QuantizationResult:
    """Return the quantized graph for ``target`` plus its honest report.

    ``activation``/``calibration`` override the drift check's scheme/grid.
    """
    spec, graph, weight_name, requested = _resolve_inputs(
        graph_or_spec, target, activation
    )
    ready, layers, applied, reason, drift, hook = _run_passes(
        graph, weight_name, requested, calibration, spikes
    )
    report = _report(
        spec, weight_name, applied, reason, layers, drift, hook,
        requested, spikes,
    )
    return QuantizationResult(ready, report)
