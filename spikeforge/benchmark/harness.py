"""Benchmark harness: time forward/backward passes and probe memory.

The harness returns plain, JSON-serialisable data. Every measurement is
preceded by warmup runs and seeded, and unavailable metrics are reported as
``None`` rather than raising, so a report is always produced.
"""

from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from spikeforge.benchmark import memory, timing
from spikeforge.benchmark.config import BenchmarkConfig, default_config
from spikeforge.benchmark.energy import energy_block
from spikeforge.runtime import device as device_mod
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator import input_shape
from spikeforge.simulator.compiled_step import (
    CompiledStep,
    compiled_step_available,
)
from spikeforge.simulator.execution import execute
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

FLAT_FEATURES = 28 * 28
SPIKE_PROBABILITY = 0.3
_StepFn = Optional[Callable[..., Any]]


def _spikes(
    spec: TopologySpec,
    steps: int,
    batch: int,
    device: torch.device,
    seed: int,
) -> torch.Tensor:
    """Build a seeded random spike train shaped for ``spec``'s input."""
    generator = torch.Generator().manual_seed(seed)
    flat = torch.rand(steps, batch, FLAT_FEATURES, generator=generator)
    shaped = input_shape.to_input_shape(flat, spec)
    spikes = (shaped < SPIKE_PROBABILITY).to(torch.float32)
    return spikes.to(device)


def _build(name: str, device: torch.device) -> Tuple[TopologySpec, Any]:
    """Build a registry topology and move it onto ``device``."""
    spec, module = build_topology(name)
    return spec, module.to(device)


def _execute_mode(
    module: StageModule,
    spikes: torch.Tensor,
    mode: ExecutionMode,
    step_fn: _StepFn = None,
) -> Trajectory:
    """Run one pass, using an injected compiled stepper when provided."""
    if step_fn is None:
        return run(module, spikes, mode=mode)
    return execute(module, spikes, False, False, False, mode, step_fn)


def _forward_call(
    module: StageModule,
    spikes: torch.Tensor,
    mode: ExecutionMode,
    step_fn: _StepFn = None,
) -> Callable[[], None]:
    """Return a callable running one forward pass without gradients."""

    def call() -> None:
        with torch.no_grad():
            _execute_mode(module, spikes, mode, step_fn)

    return call


def _backward_call(
    module: StageModule,
    spikes: torch.Tensor,
    mode: ExecutionMode,
    step_fn: _StepFn = None,
) -> Callable[[], None]:
    """Return a callable running one forward and backward pass."""

    def call() -> None:
        module.zero_grad(set_to_none=True)
        trajectory = _execute_mode(module, spikes, mode, step_fn)
        trajectory.logits.pow(2).mean().backward()

    return call


def _measure(
    module: StageModule,
    spikes: torch.Tensor,
    mode: ExecutionMode,
    config: BenchmarkConfig,
    device: torch.device,
    step_fn: _StepFn = None,
) -> Dict[str, Any]:
    """Time forward (and, when requested, backward) passes for one case."""
    args = (device, config.repeats, config.warmup, config.steps)
    forward = timing.time_calls(
        _forward_call(module, spikes, mode, step_fn), *args
    )
    backward = None
    if config.backward:
        backward = timing.time_calls(
            _backward_call(module, spikes, mode, step_fn), *args
        )
    return {"forward": forward, "backward": backward}


def _stepper(
    module: StageModule, mode: str
) -> Tuple[_StepFn, Optional[CompiledStep]]:
    """Return ``(step_fn, stepper)`` for the compiled mode, else eager."""
    if mode != "production_compiled":
        return None, None
    stepper = CompiledStep(module, enabled=True)
    return stepper.step, stepper


def _mode_of(mode: str) -> ExecutionMode:
    """Map a benchmark mode label to an :class:`ExecutionMode`."""
    if mode == "educational":
        return ExecutionMode.EDUCATIONAL
    return ExecutionMode.PRODUCTION


def _compile_state(stepper: Optional[CompiledStep]) -> Tuple[bool, str]:
    """Return ``(compiled, status)`` for a stepper or the eager default."""
    if stepper is None:
        return False, "eager"
    return stepper.compiled, stepper.status


def _record(
    topology: str,
    mode: str,
    stepper: Optional[CompiledStep],
    measured: Dict[str, Any],
    memory_info: Dict[str, Any],
) -> Dict[str, Any]:
    """Assemble one JSON-ready topology/mode result."""
    compiled, status = _compile_state(stepper)
    return {
        "topology": topology,
        "mode": mode,
        "compiled": compiled,
        "compile_status": status,
        "forward": measured["forward"],
        "backward": measured["backward"],
        "memory": memory_info,
    }


def _case(
    topology: str, mode: str, config: BenchmarkConfig, device: torch.device
) -> Dict[str, Any]:
    """Measure one topology/mode combination and return its record."""
    torch.manual_seed(config.seed)
    spec, module = _build(topology, device)
    spikes = _spikes(
        spec, config.steps, config.batch_size, device, config.seed
    )
    step_fn, stepper = _stepper(module, mode)
    core = _mode_of(mode)
    measured = _measure(module, spikes, core, config, device, step_fn)
    memory_info = memory.snapshot(
        _forward_call(module, spikes, core, step_fn), device
    )
    record = _record(topology, mode, stepper, measured, memory_info)
    record["energy"] = energy_block(module, spikes, config)
    return record


def _modes(config: BenchmarkConfig) -> List[str]:
    """Return the mode labels measured for ``config``."""
    modes = ["production", "educational"]
    if config.compiled:
        modes.append("production_compiled")
    return modes


def _cuda_name() -> Optional[str]:
    """Return the current CUDA device name, or None without a GPU."""
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(0)


def _environment(device: torch.device) -> Dict[str, Any]:
    """Describe the torch build and resolved device for the report."""
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": _cuda_name(),
        "compile_available": compiled_step_available(),
    }


def _report(
    config: BenchmarkConfig,
    device: torch.device,
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble the top-level JSON-ready benchmark report."""
    return {
        "config": asdict(config),
        "environment": _environment(device),
        "results": results,
    }


def run_benchmark(
    config: Optional[BenchmarkConfig] = None,
) -> Dict[str, Any]:
    """Measure topologies and modes, returning JSON-able data.

    The result holds ``config``, ``environment`` (torch build and device),
    and ``results`` (one record per topology and mode, each with forward,
    backward, and memory blocks).
    """
    cfg = config if config is not None else default_config()
    device = device_mod.resolve(cfg.device)
    results = [
        _case(topology, mode, cfg, device)
        for topology in cfg.topologies
        for mode in _modes(cfg)
    ]
    return _report(cfg, device, results)
