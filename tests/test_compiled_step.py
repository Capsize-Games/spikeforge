"""Tests for the opt-in ``torch.compile`` step wrapper and its fallback."""

import functools
from typing import Any, Callable

import pytest
import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator.compiled_step import (
    STATUS_COMPILED,
    STATUS_EAGER,
    STATUS_FALLBACK,
    STATUS_UNAVAILABLE,
    CompiledStep,
    compiled_step_available,
)
from spikeforge.simulator.production import run_production
from spikeforge.simulator.runner import run
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.stage_module import StageModule


def _module() -> StageModule:
    """Return a tiny ``fc_legacy`` module for the fallback tests."""
    return build_module(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=5)
    )


def _spikes() -> torch.Tensor:
    """Return a fixed tiny spike train."""
    torch.manual_seed(0)
    return torch.rand(3, 2, 5)


def _raise_compile(_fn: Any) -> Callable[..., Any]:
    """Fail at compile time, standing in for an unusable compiler."""
    raise RuntimeError("compiler unavailable")


def _explode(_fn: Any) -> Callable[..., Any]:
    """Return a callable that raises, standing in for a failed graph."""

    def broken(_x: Any, _state: Any) -> Any:
        raise RuntimeError("graph execution failed")

    return broken


def test_disabled_wrapper_is_eager() -> None:
    """A disabled wrapper reports eager and does not compile."""
    stepper = CompiledStep(_module(), enabled=False)
    assert stepper.status == STATUS_EAGER
    assert stepper.compiled is False


def test_missing_compiler_is_unavailable() -> None:
    """An explicit ``None`` compiler reports unavailable and still runs."""
    result = run_production(_module(), _spikes(), compiled=True, compiler=None)
    assert result.compiled is False
    assert result.status == STATUS_UNAVAILABLE
    assert not result.trajectory.recorded


def test_compile_error_falls_back() -> None:
    """A compiler that raises degrades to the eager path."""
    stepper = CompiledStep(_module(), enabled=True, compiler=_raise_compile)
    assert stepper.status == STATUS_FALLBACK
    assert stepper.compiled is False


def test_runtime_failure_falls_back_to_eager() -> None:
    """A compiled callable that raises on first use falls back and matches."""
    module = _module()
    spikes = _spikes()
    eager = run(module, spikes, mode=ExecutionMode.PRODUCTION)
    result = run_production(module, spikes, compiled=True, compiler=_explode)
    assert result.status == STATUS_FALLBACK
    assert result.compiled is False
    assert torch.allclose(eager.logits, result.logits, atol=1e-6)
    assert not result.trajectory.recorded


def test_auto_compiler_reports_its_availability() -> None:
    """The default compiler resolves to compiled when torch supports it."""
    stepper = CompiledStep(_module(), enabled=True)
    if compiled_step_available():
        assert stepper.status == STATUS_COMPILED
        assert stepper.compiled is True
    else:
        assert stepper.status == STATUS_UNAVAILABLE
        assert stepper.compiled is False


@pytest.mark.skipif(
    not compiled_step_available(), reason="torch.compile is unavailable"
)
def test_dynamo_compiled_forward_matches_eager() -> None:
    """A dynamo-compiled production forward matches the eager logits."""
    torch.manual_seed(0)
    module = build_module(
        presets.fc_small(hidden=4, beta=0.9, num_classes=3, input_size=5)
    )
    spikes = torch.rand(3, 2, 5)
    eager = run(module, spikes, mode=ExecutionMode.PRODUCTION)
    compiler = functools.partial(torch.compile, backend="eager")
    result = run_production(module, spikes, compiled=True, compiler=compiler)
    assert result.compiled is True
    assert result.status == STATUS_COMPILED
    assert torch.allclose(eager.logits, result.logits, atol=1e-6)
    assert not result.trajectory.recorded
