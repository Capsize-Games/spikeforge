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
device-quantized result is never claimed. The weight pass lives in
:mod:`spikeforge_targets.quantize_weights`.
"""

from typing import Any, Dict, Optional, Tuple

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.activation_quant import Calibration, refused_report
from spikeforge_targets.activation_quant_graph import (
    GraphActivationQuantizer,
    calibrate_graph,
)
from spikeforge_targets.quantize_report import Layers, QuantizationReport
from spikeforge_targets.quantize_result import QuantizationResult
from spikeforge_targets.quantize_weights import (
    NO_QUANTIZATION,
    UNKNOWN_SCHEME,
    _weights,
)
from spikeforge_targets.registry import get_target
from spikeforge_targets.rewrite_drift import rewrite_drift
from spikeforge_targets.target_spec import TargetSpec

__all__ = [
    "ACTIVATION_KEY",
    "FIXTURE_SOURCE",
    "NO_FIXTURE",
    "NO_QUANTIZATION",
    "UNKNOWN_SCHEME",
    "WEIGHT_KEY",
    "WEIGHTS",
    "declared_report",
    "quantize",
]

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
