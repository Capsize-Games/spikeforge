"""The single temporal loop that executes a topology step by step.

Both execution modes and both step paths (eager and compiled) funnel through
:func:`execute`, so ``PRODUCTION`` versus ``EDUCATIONAL`` is a recording flag
on one shared loop rather than a fork. The per-step ``step_fn`` is injected so
the same loop drives an eager ``StageModule`` or a compiled wrapper.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

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
from snn_interpreter.topology.stage_module import CURRENT_KEY, StageModule

_FrameMap = Dict[str, List[torch.Tensor]]
_StepOutputs = Tuple[Dict[str, torch.Tensor], Dict[str, Any]]
_StepFn = Callable[[torch.Tensor, Optional[Dict[str, Any]]], _StepOutputs]


def _captures(
    track: bool, membrane: bool, current: bool, mode: ExecutionMode
) -> Tuple[bool, bool, bool]:
    """Resolve which traces to record from the flags and execution mode."""
    full = mode is ExecutionMode.EDUCATIONAL
    return track or full, membrane or full, current or full


def _frame_map(names: List[str], flag: bool) -> _FrameMap:
    """Return a per-stage frame accumulator when ``flag`` is set."""
    return {name: [] for name in names} if flag else {}


def _recorders(
    spec: TopologySpec,
    track: bool,
    membrane: bool,
    current: bool,
    mode: ExecutionMode,
) -> Tuple[_FrameMap, _FrameMap, _FrameMap]:
    """Create per-stage frame accumulators for the requested traces."""
    record = _captures(track, membrane, current, mode)
    names = list(neuron_kinds(spec))
    spike_frames = _frame_map(names, record[0])
    mem_frames = _frame_map(names, record[1])
    cur_frames = _frame_map(names, record[2])
    return spike_frames, mem_frames, cur_frames


def _record(
    outputs: Dict[str, torch.Tensor],
    state: Dict[str, Any],
    spikes: _FrameMap,
    membranes: _FrameMap,
    currents: _FrameMap,
) -> None:
    """Append this step's spike, membrane, and current frames."""
    for name in spikes:
        spikes[name].append(outputs[name])
    for name in membranes:
        membranes[name].append(membrane_of(state[name]))
    live = state.get(CURRENT_KEY, {})
    for name in currents:
        currents[name].append(live[name])


def _stack(frames: _FrameMap) -> Dict[str, torch.Tensor]:
    """Stack per-step frame lists into ``[T, ...]`` trace tensors."""
    return {name: torch.stack(items) for name, items in frames.items()}


def _simulate(
    step_fn: _StepFn,
    spikes: torch.Tensor,
    spec: TopologySpec,
    kind: str,
    spike_frames: _FrameMap,
    mem_frames: _FrameMap,
    cur_frames: _FrameMap,
) -> torch.Tensor:
    """Step over the train with ``step_fn``, recording requested frames."""
    state = initial_state(spec, spikes[0])
    total: Optional[torch.Tensor] = None
    for index in range(int(spikes.size(0))):
        frame = normalise_frame(spikes[index], kind)
        outputs, state = step_fn(frame, state)
        readout = outputs[spec.output]
        total = readout if total is None else total + readout
        _record(outputs, state, spike_frames, mem_frames, cur_frames)
    return total


def _trajectory(
    steps: int,
    total: torch.Tensor,
    spike_frames: _FrameMap,
    mem_frames: _FrameMap,
    cur_frames: _FrameMap,
) -> Trajectory:
    """Assemble the run's trajectory from its readout and frame maps."""
    return Trajectory(
        steps=steps,
        logits=total / steps,
        spikes=_stack(spike_frames),
        membranes=_stack(mem_frames),
        currents=_stack(cur_frames),
    )


def execute(
    module: StageModule,
    spikes: torch.Tensor,
    track: bool,
    membrane: bool,
    current: bool,
    mode: ExecutionMode,
    step_fn: _StepFn,
) -> Trajectory:
    """Run one temporal loop with an explicit per-step callable."""
    spec = spec_of(module)
    steps = int(spikes.size(0))
    kind = spec.stage(spec.input).kind
    recorders = _recorders(spec, track, membrane, current, mode)
    total = _simulate(step_fn, spikes, spec, kind, *recorders)
    return _trajectory(steps, total, *recorders)
