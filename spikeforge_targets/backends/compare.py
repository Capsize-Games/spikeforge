"""Compare a backend result to the reference interpreter's trajectory.

Both results carry a readout and per-node spike trains in the same shape, so
the shared drift metrics apply directly. The comparison is a genuine
cross-check: the reference side is produced by
:class:`~spikeforge.nir_bridge.interpreter.NirInterpreter`, never by the
backend SDK, so a backend-only discrepancy shows up as drift.
"""

from typing import Any, Dict, List

from spikeforge.nir_bridge import drift
from spikeforge_targets.backends.result import BackendResult


def _shared_names(
    actual: BackendResult, reference: BackendResult
) -> List[str]:
    """Return the spiking node names both results recorded."""
    return sorted(name for name in actual.spikes if name in reference.spikes)


def _spike_summary(
    actual: BackendResult, reference: BackendResult, names: List[str]
) -> Dict[str, Any]:
    """Return aggregate spike drift over the shared node names."""
    if not names:
        return {"nodes": [], "max_abs": 0.0, "agreement": 1.0}
    metrics = {
        name: drift.compare(actual.spikes[name], reference.spikes[name])
        for name in names
    }
    return {
        "nodes": names,
        "max_abs": max(item["max_abs"] for item in metrics.values()),
        "agreement": min(item["agreement"] for item in metrics.values()),
    }


def compare_results(
    actual: BackendResult, reference: BackendResult
) -> Dict[str, Any]:
    """Return readout and spike drift of ``actual`` against ``reference``.

    ``reference`` must have completed a run; ``actual`` must carry a readout.
    The readout is compared directly and the shared spike trains are folded
    into a single max-error/agreement pair.
    """
    readout = drift.compare(actual.readout, reference.readout)
    spikes = _spike_summary(
        actual, reference, _shared_names(actual, reference)
    )
    return {"readout": readout, "spikes": spikes}
