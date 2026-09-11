"""Inter-spike-interval statistics for recorded spike trains."""

from typing import Dict, List, Optional

import torch

from spikeforge.simulator.trajectory import Trajectory

#: A statistic is ``None`` when it is undefined (fewer than two spikes).
IsiStats = Dict[str, Optional[float]]


def _column_isis(column: torch.Tensor) -> List[torch.Tensor]:
    """Return the inter-spike intervals of one neuron's spike column."""
    times = torch.nonzero(column > 0).flatten()
    if times.numel() < 2:
        return []
    return [times[1:] - times[:-1]]


def isi_values(spikes: torch.Tensor) -> torch.Tensor:
    """Return every inter-spike interval across a ``[T, ...]`` tensor."""
    flat = spikes.detach().reshape(int(spikes.size(0)), -1)
    chunks: List[torch.Tensor] = []
    for index in range(flat.size(1)):
        chunks.extend(_column_isis(flat[:, index]))
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks).float()


def _empty_stats() -> IsiStats:
    """Return all-undefined statistics for a train with too few spikes."""
    return {
        "count": 0,
        "mean": None,
        "median": None,
        "std": None,
        "cv": None,
    }


def isi_stats(spikes: torch.Tensor) -> IsiStats:
    """Return count/mean/median/std/cv of a spike tensor's intervals."""
    values = isi_values(spikes)
    if values.numel() == 0:
        return _empty_stats()
    mean = float(values.mean())
    std = float(values.std(unbiased=False)) if values.numel() > 1 else 0.0
    return {
        "count": int(values.numel()),
        "mean": mean,
        "median": float(values.median()),
        "std": std,
        "cv": (std / mean) if mean else None,
    }


def stage_isi(trajectory: Trajectory) -> Dict[str, IsiStats]:
    """Return ISI statistics for every recorded neuron stage."""
    return {
        name: isi_stats(trace)
        for name, trace in trajectory.spikes.items()
    }
