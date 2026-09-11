"""Readout parity between a sparse run and the dense trajectory.

The sparse path is only credible if it reproduces the dense readout, so this
module owns the tolerance comparison the acceptance bar asserts. Op counts and
the sparse efficiency ratio travel alongside the parity result so a caller has
the whole event-driven story in one JSON-able block.
"""

from typing import Any, Dict

from spikeforge.simulator.trajectory import Trajectory
from spikeforge_targets.event_runtime.sparse_result import SparseResult

#: Default absolute tolerance for sparse-vs-dense readout parity.
DEFAULT_TOLERANCE = 1e-4


def readout_delta(sparse: SparseResult, dense: Trajectory) -> float:
    """Return the max-abs difference between the two readouts."""
    if sparse.trajectory.logits.shape != dense.logits.shape:
        raise ValueError("readout shapes differ between sparse and dense")
    diff = (sparse.trajectory.logits - dense.logits).abs()
    return float(diff.max())


def matches(
    sparse: SparseResult,
    dense: Trajectory,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """Return True when the sparse readout is within ``tolerance``."""
    return readout_delta(sparse, dense) <= tolerance


def compare(
    sparse: SparseResult,
    dense: Trajectory,
    tolerance: float = DEFAULT_TOLERANCE,
) -> Dict[str, Any]:
    """Return a JSON-able parity, op-count, and efficiency comparison."""
    delta = readout_delta(sparse, dense)
    mac = sparse.mac
    return {
        "max_abs": delta,
        "tolerance": float(tolerance),
        "within_tolerance": delta <= tolerance,
        "ops": dict(sparse.counts),
        "efficiency": {"sop_over_mac": None if mac == 0 else sparse.sop / mac},
        "density": sparse.density,
    }
