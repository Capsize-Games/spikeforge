"""Firing-rate histogram data (bin edges and counts) for a spike tensor."""

from typing import Any, Dict, List

import torch

from spikeforge.simulator.trajectory import Trajectory

#: ``{"edges": [...], "counts": [...]}`` for a future chart.
Histogram = Dict[str, List[Any]]


def firing_rate_histogram(
    spikes: torch.Tensor, bins: int = 10
) -> Histogram:
    """Return bin edges and counts of per-neuron firing rates.

    Each neuron's firing rate is its mean spike value over time; those
    rates are histogrammed into ``bins`` equal-width bins. The result is a
    plain ``{"edges": [...], "counts": [...]}`` pair carrying no tensors.
    """
    if spikes.numel() == 0:
        return {"edges": [], "counts": []}
    flat = spikes.detach().reshape(int(spikes.size(0)), -1).float()
    counts, edges = torch.histogram(flat.mean(dim=0), bins=bins)
    return {"edges": edges.tolist(), "counts": counts.tolist()}


def stage_histograms(
    trajectory: Trajectory, bins: int = 10
) -> Dict[str, Histogram]:
    """Return the firing-rate histogram of every recorded neuron stage."""
    return {
        name: firing_rate_histogram(trace, bins)
        for name, trace in trajectory.spikes.items()
    }
