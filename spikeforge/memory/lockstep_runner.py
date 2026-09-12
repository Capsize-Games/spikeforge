"""Drive a frozen base network and an associative memory in lockstep."""

from typing import Tuple

import torch

from spikeforge.memory.one_shot_associative_memory import (
    OneShotAssociativeMemory,
)
from spikeforge.network.inference import hidden_stage
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory


def hidden_trace(
    net: torch.nn.Module, spikes: torch.Tensor,
) -> Tuple[torch.Tensor, Trajectory]:
    """Return ``([T, B, H]`` hidden spikes, full trajectory)`` for ``net``.

    Reuses :func:`spikeforge.network.inference.hidden_stage` to locate
    the neuron stage feeding the output, so this works for any
    topology rather than hard-coding a stage name.
    """
    trajectory = run(net, spikes, track=True)
    stage = hidden_stage(spec_of(net))
    return trajectory.spikes[stage], trajectory


def combined_predictions(
    net: torch.nn.Module,
    memory: OneShotAssociativeMemory,
    spikes: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B]`` predicted class indices over base + memory classes.

    Base-class indices ``0..num_base_classes-1`` come first, followed
    by one index per taught memory class, in teaching order. The
    caller maps memory indices back to real labels.
    """
    hidden, trajectory = hidden_trace(net, spikes)
    memory.reset_state(hidden.size(1))
    with torch.no_grad():
        mem_spikes = torch.stack(
            [memory.step(hidden[t]) for t in range(hidden.size(0))]
        )
    base_counts = trajectory.spikes[spec_of(net).output].sum(dim=0)
    mem_counts = mem_spikes.sum(dim=0)
    return torch.cat([base_counts, mem_counts], dim=1).argmax(dim=1)
