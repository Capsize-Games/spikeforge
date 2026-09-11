"""Ergonomic production-mode entry point with optional compilation."""

from typing import Any

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator.compiled_step import AUTO_COMPILER, CompiledStep
from spikeforge.simulator.execution import execute
from spikeforge.simulator.production_result import ProductionResult
from spikeforge.topology.stage_module import StageModule


def run_production(
    module: StageModule,
    spikes: torch.Tensor,
    compiled: bool = False,
    compiler: Any = AUTO_COMPILER,
) -> ProductionResult:
    """Run ``module`` in production mode and report its compile status.

    Production mode records no per-step traces. ``compiled`` is opt-in; when
    compilation is unavailable or fails the run transparently uses the eager
    path, and the result's ``compiled``/``status`` say which path ran.
    """
    stepper = CompiledStep(module, enabled=compiled, compiler=compiler)
    trajectory = execute(
        module, spikes, False, False, False, ExecutionMode.PRODUCTION,
        stepper.step,
    )
    return ProductionResult(trajectory, stepper.compiled, stepper.status)
