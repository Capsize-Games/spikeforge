"""The public temporal entry point that executes a topology step by step."""

from typing import Any, Optional

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator.compiled_step import CompiledStep
from spikeforge.simulator.execution import execute
from spikeforge.simulator.grad_policy import GradPolicy
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.stage_module import StageModule


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
    grad_checkpoint: bool = False,
    bptt_steps: Optional[int] = None,
) -> Trajectory:
    """Run ``module`` over a ``[T, ...]`` spike train in one time loop.

    ``track``/``membrane``/``current`` select spike, membrane, and input
    current recording, and ``EDUCATIONAL`` records all three. ``compiled``,
    ``grad_checkpoint``, and ``bptt_steps`` are all opt-in and default off:
    left unset they leave forward values, gradients, and numerics unchanged.
    ``grad_checkpoint`` recomputes per-step activations in the backward pass;
    ``bptt_steps`` detaches the carried state every N steps (gradient horizon
    only, never forward values).
    """
    step_fn = _step_fn(module, compiled)
    policy = GradPolicy(grad_checkpoint, bptt_steps)
    return execute(
        module, spikes, track, membrane, current, mode, step_fn, policy
    )
