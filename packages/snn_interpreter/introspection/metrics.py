"""One JSON-serialisable metrics entry point for a :class:`Trajectory`."""

from typing import Any, Dict

from snn_interpreter.introspection.firing_rate import stage_firing_rates
from snn_interpreter.introspection.histogram import stage_histograms
from snn_interpreter.introspection.isi import stage_isi
from snn_interpreter.introspection.sparsity import stage_sparsity
from snn_interpreter.simulator.trajectory import Trajectory


def _stage_metrics(
    name: str,
    rates: Dict[str, float],
    spars: Dict[str, float],
    isi: Dict[str, Any],
    hist: Dict[str, Any],
) -> Dict[str, Any]:
    """Combine the metrics of one stage into a JSON-able record."""
    return {
        "firing_rate": rates[name],
        "sparsity": spars[name],
        "isi": isi[name],
        "histogram": hist[name],
    }


def trajectory_metrics(
    trajectory: Trajectory, bins: int = 10
) -> Dict[str, Any]:
    """Return firing-rate, sparsity, ISI, and histogram metrics per stage.

    The result holds only ``int``/``float``/``None``/``list``/``dict`` values,
    so it passes straight through ``json.dumps`` with no tensor conversion.
    """
    rates = stage_firing_rates(trajectory)
    spars = stage_sparsity(trajectory)
    isi = stage_isi(trajectory)
    hist = stage_histograms(trajectory, bins)
    return {
        "steps": int(trajectory.steps),
        "stages": {
            name: _stage_metrics(name, rates, spars, isi, hist)
            for name in rates
        },
    }
