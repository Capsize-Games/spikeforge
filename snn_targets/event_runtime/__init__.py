"""Sparse/event-driven execution path beside the dense simulator unroll.

Nothing here changes the dense default: ``sparse_run`` is a parallel path
that propagates spike events, and the dense ``run`` stays the untouched
back-compatible entry point. The tally it returns feeds the energy accounting
in :mod:`snn_targets.energy`.
"""

from snn_targets.event_runtime.counters import SynapticCounter
from snn_targets.event_runtime.dense_compare import (
    DEFAULT_TOLERANCE,
    compare,
    matches,
    readout_delta,
)
from snn_targets.event_runtime.errors import (
    SparseError,
    UnsupportedKindError,
)
from snn_targets.event_runtime.sparse_result import SparseResult
from snn_targets.event_runtime.sparse_runner import sparse_run
from snn_targets.event_runtime.spike_view import (
    SparseSpikes,
    synthetic_spikes,
)

__all__ = [
    "DEFAULT_TOLERANCE",
    "SparseError",
    "SparseResult",
    "SparseSpikes",
    "SynapticCounter",
    "UnsupportedKindError",
    "compare",
    "matches",
    "readout_delta",
    "sparse_run",
    "synthetic_spikes",
]
