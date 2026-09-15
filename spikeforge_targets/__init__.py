"""Deployment targets, their declared capabilities, and reports.

This package declares what each target *can* run (a :class:`TargetSpec`),
resolves whether it is actually installed through one isolated probe, and
classifies a graph node-by-node against a target. The reference interpreter
is always available; every other entry is an honest placeholder until its
optional SDK extra is installed. No backend SDK is imported at import time.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version

from spikeforge_targets.capability_matrix import (
    classify,
    classify_by_name,
    compare_targets,
)
from spikeforge_targets.matrix_result import CapabilityMatrix
from spikeforge_targets.primitives import EMITTED_PRIMITIVES
from spikeforge_targets.registry import (
    available,
    available_names,
    get_target,
    target_names,
    unavailable_names,
)
from spikeforge_targets.report import deployment_report
from spikeforge_targets.substitution import Substitution
from spikeforge_targets.summary import target_summaries, target_summary
from spikeforge_targets.target_spec import TargetKind, TargetSpec

try:
    #: Read from installed metadata so it cannot drift from the wheel.
    __version__ = _distribution_version("spikeforge-targets")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
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
