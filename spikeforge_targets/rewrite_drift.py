"""Measure how far a rewritten graph moved from the graph it replaced.

Both graphs are executed by the independent reference interpreter on the same
spike fixture and compared with the shared drift metrics. A substitution that
is not exactly faithful (for example the leak-free ``IF`` -> ``LIF``
approximation) is therefore quantified in the report rather than hidden, and
the spikes that were preserved are compared node by node.
"""

from typing import Any, Dict, List, Mapping

from spikeforge.nir_bridge import drift
from spikeforge.nir_bridge.interpreter import NirInterpreter

#: Largest readout error a rewrite may introduce and stay "within tolerance".
READOUT_MAX_ABS = 1e-4
#: Largest per-spike-train error a rewrite may introduce.
SPIKE_MAX_ABS = 1e-4
#: Lowest spike-placement agreement a rewrite may keep.
SPIKE_AGREEMENT = 0.99


def _shared(before: Mapping[str, Any], after: Mapping[str, Any]) -> List[str]:
    """Return the spiking node names present in both trajectories."""
    return sorted(name for name in before if name in after)


def _spike_summary(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the aggregate spike drift over the shared spiking nodes."""
    names = _shared(before, after)
    if not names:
        return {
            "nodes": [],
            "max_abs": 0.0,
            "mean_abs": 0.0,
            "agreement": 1.0,
        }
    metrics = {
        name: drift.compare(after[name], before[name]) for name in names
    }
    return {
        "nodes": names,
        "max_abs": max(item["max_abs"] for item in metrics.values()),
        "mean_abs": max(item["mean_abs"] for item in metrics.values()),
        "agreement": min(item["agreement"] for item in metrics.values()),
    }


def _within(readout: Mapping[str, float], spikes: Mapping[str, Any]) -> bool:
    """Return True when every measured quantity stays within tolerance."""
    return (
        readout["max_abs"] <= READOUT_MAX_ABS
        and spikes["max_abs"] <= SPIKE_MAX_ABS
        and spikes["agreement"] >= SPIKE_AGREEMENT
    )


def rewrite_drift(graph: Any, rewritten: Any, spikes: Any) -> Dict[str, Any]:
    """Return the drift of ``rewritten`` against ``graph`` over ``spikes``.

    Both graphs are run by :class:`NirInterpreter`; the readout is compared
    directly and the shared spike trains are aggregated. ``within_tolerance``
    folds the two error checks into the single verdict the report carries.
    """
    before = NirInterpreter(graph).run(spikes)
    after = NirInterpreter(rewritten).run(spikes)
    readout = drift.compare(after.readout, before.readout)
    spike = _spike_summary(before.spikes, after.spikes)
    return {
        "steps": int(before.steps),
        "readout": readout,
        "spikes": spike,
        "within_tolerance": bool(_within(readout, spike)),
    }
