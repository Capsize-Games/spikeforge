"""Produce a JSON-serialisable deployment report for one target."""

from typing import Any, Dict, List, Mapping, Optional

from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.capability_matrix import classify
from spikeforge_targets.matrix_result import CapabilityMatrix
from spikeforge_targets.registry import get_target

#: Note added when a target's enabling SDK is not installed.
UNAVAILABLE_NOTE = "target SDK is not installed; run the enabling extra"
#: Note added when a validation section accompanies the report.
VALIDATION_NOTE = "validation compares snnTorch against the exported graph"
#: Reason recorded when a spec arrives without weights to quantize.
NO_MODULE_REASON = "no built module supplied; quantization not applied"


def _validation(
    spec: TopologySpec,
    module: Optional[Any],
    spikes: Optional[Any],
    tolerances: Optional[Mapping[str, float]],
) -> Optional[Any]:
    """Return the validation report when its inputs are available.

    Imported lazily because the validator pulls in torch and the simulator,
    which the pure capability/registry queries never need.
    """
    if module is None or spikes is None:
        return None
    from spikeforge.nir_bridge.validator import validate

    return validate(spec, module, spikes, tolerances)


def _notes(
    matrix: CapabilityMatrix, validation: Optional[Any]
) -> List[str]:
    """Return the human-readable notes accompanying a report."""
    notes: List[str] = []
    if not matrix.available:
        notes.append(UNAVAILABLE_NOTE)
    if validation is not None:
        notes.append(VALIDATION_NOTE)
    return notes


def _rewrite_graph(
    graph_or_spec: Any, module: Optional[Any]
) -> Any:
    """Return the graph a rewrite should run on, copying module weights."""
    if isinstance(graph_or_spec, TopologySpec) and module is not None:
        from spikeforge.nir_bridge.exporter import to_nir

        return to_nir(graph_or_spec, module)
    return graph_or_spec


def _quantization(
    graph_or_spec: Any, module: Optional[Any], target_name: str,
    spikes: Optional[Any], activation: Optional[str],
) -> Dict[str, Any]:
    """Return the quantization section, applied only when weights exist.

    Declared-unapplied when a spec has no built module; imported lazily so
    a capability-only report stays free of nir and torch.
    """
    from spikeforge_targets.quantize import declared_report, quantize

    if isinstance(graph_or_spec, TopologySpec) and module is None:
        return declared_report(target_name, NO_MODULE_REASON).to_dict()
    try:
        graph = _rewrite_graph(graph_or_spec, module)
        result = quantize(graph, target_name, spikes, activation=activation)
        return result.report.to_dict()
    except Exception as exc:
        return {"target": target_name, "error": f"{type(exc).__name__}: {exc}"}


def _rewrite(
    graph_or_spec: Any, module: Optional[Any], target_name: str,
    spikes: Optional[Any],
) -> Optional[Any]:
    """Return the executed-substitution section, or ``None`` without spikes.

    Imported lazily; guarded so a rewrite failure is reported as a named
    error rather than dropped or raised.
    """
    if spikes is None:
        return None
    from spikeforge_targets.rewrite import rewrite

    graph = _rewrite_graph(graph_or_spec, module)
    try:
        return rewrite(graph, target_name, spikes).report.to_dict()
    except Exception as exc:
        return {"target": target_name, "error": f"{type(exc).__name__}: {exc}"}


def _node_buckets(matrix: CapabilityMatrix) -> Dict[str, Any]:
    """Return the supported/unsupported/substituted node buckets."""
    return {
        "supported": list(matrix.supported),
        "unsupported": list(matrix.unsupported),
        "substituted": [item.to_dict() for item in matrix.substituted],
        "counts": matrix.counts(),
    }


def _resolve_validation(
    graph_or_spec: Any, module: Optional[Any], spikes: Optional[Any],
    tolerances: Optional[Mapping[str, float]],
) -> Optional[Any]:
    """Return the validation report when ``graph_or_spec`` is a spec."""
    if not isinstance(graph_or_spec, TopologySpec):
        return None
    return _validation(graph_or_spec, module, spikes, tolerances)


def _report_fields(
    target: Any, matrix: CapabilityMatrix, validation: Optional[Any]
) -> Dict[str, Any]:
    """Return the capability-derived fields of the deployment report."""
    return {
        "target": target.to_dict(),
        "available": matrix.available,
        "deployable": matrix.deployable(),
        "nodes": _node_buckets(matrix),
        "constraints": dict(target.constraints),
        "validation": validation,
        "notes": _notes(matrix, validation),
    }


def deployment_report(
    graph_or_spec: Any, target_name: str, module: Optional[Any] = None,
    spikes: Optional[Any] = None,
    tolerances: Optional[Mapping[str, float]] = None,
    activation: Optional[str] = None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable deployment report for ``target_name``.

    An unavailable target is marked false rather than raising;
    ``activation`` overrides the declared quantization drift scheme.
    """
    target = get_target(target_name)
    matrix = classify(graph_or_spec, target)
    validation = _resolve_validation(graph_or_spec, module, spikes, tolerances)
    report = _report_fields(target, matrix, validation)
    report["quantization"] = _quantization(
        graph_or_spec, module, target_name, spikes, activation
    )
    report["rewrite"] = _rewrite(graph_or_spec, module, target_name, spikes)
    return report
