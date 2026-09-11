"""Per-stage firing-rate metrics derived from a :class:`Trajectory`."""

from typing import Dict

import torch

from spikeforge.simulator.trajectory import Trajectory


def firing_rate(spikes: torch.Tensor) -> float:
    """Return mean spikes per neuron per step of a ``[T, ...]`` tensor."""
    if spikes.numel() == 0:
        return 0.0
    return float(spikes.detach().float().mean())


def stage_firing_rates(trajectory: Trajectory) -> Dict[str, float]:
    """Return the firing rate of every recorded neuron stage."""
    return {
        name: firing_rate(trace)
        for name, trace in trajectory.spikes.items()
    }
