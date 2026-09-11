"""Per-stage sparsity metrics derived from a :class:`Trajectory`."""

from typing import Dict

import torch

from spikeforge.simulator.trajectory import Trajectory


def sparsity(spikes: torch.Tensor) -> float:
    """Return the fraction of zero entries in a ``[T, ...]`` spike tensor."""
    if spikes.numel() == 0:
        return 0.0
    return float((spikes.detach() == 0).float().mean())


def stage_sparsity(trajectory: Trajectory) -> Dict[str, float]:
    """Return the sparsity of every recorded neuron stage."""
    return {
        name: sparsity(trace)
        for name, trace in trajectory.spikes.items()
    }
