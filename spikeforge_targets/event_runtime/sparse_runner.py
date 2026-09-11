"""Sparse/event-driven execution path beside the dense simulator unroll.

This is an inference-and-counting path: it propagates spikes through the same
topology as :func:`~spikeforge.simulator.runner.run` but multiplies only
active entries, so the readout matches the dense path while the synaptic
operation count drops with input sparsity. It is training-free by design.
"""

from typing import Optional, Tuple

import torch

from spikeforge.simulator.frames import normalise_frame
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.state import initial_state
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.stage_module import StageModule
from spikeforge_targets.event_runtime.counters import SynapticCounter
from spikeforge_targets.event_runtime.sparse_result import SparseResult
from spikeforge_targets.event_runtime.sparse_step import SparseStageModule
from spikeforge_targets.event_runtime.spike_view import SparseSpikes


def _loop(
    module: StageModule, spikes: torch.Tensor,
    counter: Optional[SynapticCounter],
) -> Tuple[torch.Tensor, int]:
    """Step the sparse module over time, returning the summed readout."""
    spec = spec_of(module)
    kind = spec.stage(spec.input).kind
    stepper = SparseStageModule(module, counter)
    state = initial_state(spec, spikes[0])
    total: Optional[torch.Tensor] = None
    for index in range(int(spikes.size(0))):
        frame = normalise_frame(spikes[index], kind)
        outputs, state = stepper.step(frame, state)
        readout = outputs[spec.output]
        total = readout if total is None else total + readout
        if counter is not None:
            counter.tick()
    if total is None:
        raise ValueError("sparse_run needs at least one timestep")
    return total, int(spikes.size(0))


def sparse_run(
    module: StageModule,
    spikes: torch.Tensor,
    counters: bool = True,
) -> SparseResult:
    """Run ``module`` over a ``[T, ...]`` spike train, propagating events only.

    ``counters`` attaches the SOP/MAC/AC/timestep tally (default on). The
    result's trajectory is a drop-in for the dense readout, so a caller can
    compare the two within tolerance while reading the reduced op count.
    """
    counter = SynapticCounter() if counters else None
    total, steps = _loop(module, spikes, counter)
    return SparseResult(
        trajectory=Trajectory(steps=steps, logits=total / steps),
        counts={} if counter is None else counter.counts(),
        per_stage={} if counter is None else counter.stages(),
        density=SparseSpikes.from_dense(spikes).density,
    )
