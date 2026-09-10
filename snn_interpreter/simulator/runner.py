"""The public temporal entry point that executes a topology step by step."""

from typing import Any

import torch

from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.compiled_step import CompiledStep
from snn_interpreter.simulator.execution import execute
from snn_interpreter.simulator.trajectory import Trajectory
from snn_interpreter.topology.stage_module import StageModule


def _step_fn(module: StageModule, compiled: bool) -> Any:
    """Return the per-step callable, compiling it when requested."""
    if not compiled:
        return module.step
    return CompiledStep(module, enabled=True).step


def run(
    module: StageModule,
    spikes: torch.Tensor,
    track: bool = False,
    membrane: bool = False,
    current: bool = False,
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
    compiled: bool = False,
) -> Trajectory:
    """Run ``module`` over a ``[T, ...]`` spike train in one time loop.

    ``track``/``membrane``/``current`` select spike, membrane, and input
    current recording, and ``EDUCATIONAL`` records all three. ``compiled`` is
    opt-in, defaults off, and leaves every default unchanged.
    """
    step_fn = _step_fn(module, compiled)
    return execute(
        module, spikes, track, membrane, current, mode, step_fn
    )
