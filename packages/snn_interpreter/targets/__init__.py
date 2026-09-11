"""Deployment targets, their declared capabilities, and reports.

This package declares what each target *can* run (a :class:`TargetSpec`),
resolves whether it is actually installed through one isolated probe, and
classifies a graph node-by-node against a target. The reference interpreter
is always available; every other entry is an honest placeholder until its
optional SDK extra is installed. No backend SDK is imported at import time.
"""

from snn_interpreter.targets.capability_matrix import (
    classify,
    classify_by_name,
    compare_targets,
)
from snn_interpreter.targets.matrix_result import CapabilityMatrix
from snn_interpreter.targets.primitives import EMITTED_PRIMITIVES
from snn_interpreter.targets.registry import (
    available,
    available_names,
    get_target,
    target_names,
    unavailable_names,
)
from snn_interpreter.targets.report import deployment_report
from snn_interpreter.targets.substitution import Substitution
from snn_interpreter.targets.summary import target_summaries, target_summary
from snn_interpreter.targets.target_spec import TargetKind, TargetSpec

__all__ = [
    "EMITTED_PRIMITIVES",
    "CapabilityMatrix",
    "Substitution",
    "TargetKind",
    "TargetSpec",
    "available",
    "available_names",
    "classify",
    "classify_by_name",
    "compare_targets",
    "deployment_report",
    "get_target",
    "target_names",
    "target_summaries",
    "target_summary",
    "unavailable_names",
]
