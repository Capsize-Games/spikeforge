"""The single temporal loop that executes a topology step by step."""

from typing import Any, Dict, List, Optional, Tuple

import torch

from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.frames import normalise_frame
from snn_interpreter.simulator.module_spec import spec_of
from snn_interpreter.simulator.state import (
    initial_state,
    membrane_of,
    neuron_kinds,
)
from snn_interpreter.simulator.trajectory import Trajectory
from snn_interpreter.topology.spec import TopologySpec
from snn_interpreter.topology.stage_module import StageModule

_FrameMap = Dict[str, List[torch.Tensor]]


def _captures(
    track: bool, membrane: bool, mode: ExecutionMode
) -> Tuple[bool, bool]:
    """Resolve which traces to record from the flags and execution mode."""
    full = mode is ExecutionMode.EDUCATIONAL
    return track or full, membrane or full


def _recorders(
    spec: TopologySpec, track: bool, membrane: bool, mode: ExecutionMode
) -> Tuple[_FrameMap, _FrameMap]:
    """Create per-stage frame accumulators for the requested traces."""
    record_spikes, record_membranes = _captures(track, membrane, mode)
    names = list(neuron_kinds(spec))
    return (
        {name: [] for name in names} if record_spikes else {},
        {name: [] for name in names} if record_membranes else {},
    )


def _record(
    outputs: Dict[str, torch.Tensor],
    state: Dict[str, Any],
    spikes: _FrameMap,
    membranes: _FrameMap,
) -> None:
    """Append this step's spike and membrane frames to the accumulators."""
    for name in spikes:
        spikes[name].append(outputs[name])
    for name in membranes:
        membranes[name].append(membrane_of(state[name]))


def _stack(frames: _FrameMap) -> Dict[str, torch.Tensor]:
    """Stack per-step frame lists into ``[T, ...]`` trace tensors."""
    return {name: torch.stack(items) for name, items in frames.items()}


def _simulate(
    module: StageModule,
    spikes: torch.Tensor,
    spec: TopologySpec,
    kind: str,
    spike_frames: _FrameMap,
    mem_frames: _FrameMap,
) -> torch.Tensor:
    """Step the module over the train, recording the requested frames."""
    state = initial_state(spec, spikes[0])
    total: Optional[torch.Tensor] = None
    for index in range(int(spikes.size(0))):
        frame = normalise_frame(spikes[index], kind)
        outputs, state = module.step(frame, state)
        readout = outputs[spec.output]
        total = readout if total is None else total + readout
        _record(outputs, state, spike_frames, mem_frames)
    return total


def _trajectory(
    steps: int,
    total: torch.Tensor,
    spike_frames: _FrameMap,
    mem_frames: _FrameMap,
) -> Trajectory:
    """Assemble the run's trajectory from its readout and frame maps."""
    return Trajectory(
        steps=steps,
        logits=total / steps,
        spikes=_stack(spike_frames),
        membranes=_stack(mem_frames),
    )


def run(
    module: StageModule,
    spikes: torch.Tensor,
    track: bool = False,
    membrane: bool = False,
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> Trajectory:
    """Run ``module`` over a ``[T, ...]`` spike train in one time loop.

    ``track`` records per-stage spikes, ``membrane`` additionally records
    membranes, and ``EDUCATIONAL`` mode records both. Logits are the
    ``output`` spikes averaged over ``steps``.
    """
    spec = spec_of(module)
    steps = int(spikes.size(0))
    kind = spec.stage(spec.input).kind
    spike_frames, mem_frames = _recorders(spec, track, membrane, mode)
    total = _simulate(module, spikes, spec, kind, spike_frames, mem_frames)
    return _trajectory(steps, total, spike_frames, mem_frames)
