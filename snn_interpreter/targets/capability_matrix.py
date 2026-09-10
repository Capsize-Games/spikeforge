"""Classify every node of a graph against a target's declared support."""

from typing import Any, Dict, List

from snn_interpreter.targets import probe
from snn_interpreter.targets.matrix_result import CapabilityMatrix
from snn_interpreter.targets.node_view import node_kinds
from snn_interpreter.targets.registry import get_target, target_names
from snn_interpreter.targets.substitution import Substitution
from snn_interpreter.targets.target_spec import TargetSpec


def classify(target: Any, spec: TargetSpec) -> CapabilityMatrix:
    """Return the capability matrix of ``target`` for ``spec``.

    Each node is placed in exactly one bucket: supported, substituted (when
    the target declares a replacement primitive for it), or unsupported.
    Nothing is dropped, so the buckets always partition the node set.
    """
    supported: List[str] = []
    unsupported: List[str] = []
    substituted: List[Substitution] = []
    for name, kind in node_kinds(target):
        form = spec.substitute(kind)
        if spec.supports(kind):
            supported.append(name)
        elif form is not None:
            substituted.append(Substitution(name, kind, form))
        else:
            unsupported.append(name)
    return CapabilityMatrix(
        target=spec.name,
        available=probe.extra_available(spec.extra),
        supported=tuple(supported),
        unsupported=tuple(unsupported),
        substituted=tuple(substituted),
    )


def classify_by_name(target: Any, name: str) -> CapabilityMatrix:
    """Return the capability matrix of ``target`` for target ``name``."""
    return classify(target, get_target(name))


def compare_targets(target: Any) -> Dict[str, Any]:
    """Return one JSON-able capability row per built-in target."""
    return {
        name: classify_by_name(target, name).to_dict()
        for name in target_names()
    }
