"""Bridge a spiking embedder's output spikes into memory-module calls.

Unlike :mod:`spikeforge.memory.lockstep_runner` (issue #22, a frozen
*classifier* with a hidden layer feeding memory), an episodic embedder
(issue #24) has no separate classification head: its own output stage
*is* the spike pattern memory reads.
"""

from typing import Tuple

import torch

from spikeforge.memory.one_shot_associative_memory import (
    OneShotAssociativeMemory,
)
from spikeforge.simulator import input_shape
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.stage_module import StageModule


def output_spike_trace(
    net: StageModule, spikes: torch.Tensor,
) -> Tuple[torch.Tensor, Trajectory]:
    """Return ``([T, B, D]`` output spikes, full trajectory)`` for ``net``.

    Reshaping through ``input_shape.to_input_shape`` is a no-op for a
    flat embedder (``fc_small``) and restores the spatial geometry a
    ``conv_net`` embedder needs.
    """
    spec = spec_of(net)
    shaped = input_shape.to_input_shape(spikes, spec)
    trajectory = run(net, shaped, track=True)
    return trajectory.spikes[spec.output], trajectory


def teach_from_example(
    memory: OneShotAssociativeMemory,
    net: StageModule,
    spikes: torch.Tensor,
) -> int:
    """One-shot teach ``memory`` from one example's encoded ``spikes``."""
    trace, _ = output_spike_trace(net, spikes)
    return memory.teach(trace[:, 0, :])


def memory_predictions(
    memory: OneShotAssociativeMemory,
    net: StageModule,
    spikes: torch.Tensor,
) -> torch.Tensor:
    """Return ``[B]`` predicted class indices from memory alone."""
    trace, _ = output_spike_trace(net, spikes)
    memory.reset_state(trace.size(1))
    counts = torch.zeros(trace.size(1), memory.num_classes)
    for step in trace:
        counts += memory.step(step)
    return counts.argmax(dim=1)
