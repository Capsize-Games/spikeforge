"""Produce a JSON-serialisable deployment report for one target."""

from typing import Any, Dict, List, Mapping, Optional

from snn_interpreter.targets.capability_matrix import classify
from snn_interpreter.targets.matrix_result import CapabilityMatrix
from snn_interpreter.targets.registry import get_target
from snn_interpreter.topology.spec import TopologySpec

#: Note added when a target's enabling SDK is not installed.
UNAVAILABLE_NOTE = "target SDK is not installed; run the enabling extra"
#: Note added when a validation section accompanies the report.
VALIDATION_NOTE = "validation compares snnTorch against the exported graph"


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
    from snn_interpreter.nir_bridge.validator import validate

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


def _node_buckets(matrix: CapabilityMatrix) -> Dict[str, Any]:
    """Return the supported/unsupported/substituted node buckets."""
    return {
        "supported": list(matrix.supported),
        "unsupported": list(matrix.unsupported),
        "substituted": [item.to_dict() for item in matrix.substituted],
        "counts": matrix.counts(),
    }


def deployment_report(
    graph_or_spec: Any,
    target_name: str,
    module: Optional[Any] = None,
    spikes: Optional[Any] = None,
    tolerances: Optional[Mapping[str, float]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serialisable deployment report for ``target_name``.

    The report is always produced; an unavailable target is marked
    ``available``/``deployable`` false rather than raising, leaving
    user-facing messaging to a later phase. When ``graph_or_spec`` is a
    :class:`TopologySpec` and both ``module`` and ``spikes`` are supplied, the
    report also carries the :class:`ValidationReport` under ``validation``.
    """
    target = get_target(target_name)
    matrix = classify(graph_or_spec, target)
    validation = None
    if isinstance(graph_or_spec, TopologySpec):
        validation = _validation(graph_or_spec, module, spikes, tolerances)
    return {
        "target": target.to_dict(),
        "available": matrix.available,
        "deployable": matrix.deployable(),
        "nodes": _node_buckets(matrix),
        "constraints": dict(target.constraints),
        "validation": validation,
        "notes": _notes(matrix, validation),
    }
