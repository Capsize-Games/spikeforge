"""Norse backend: honest absence, refused graphs, and a stubbed run."""

import sys
import types
from typing import Any, Optional, Tuple

import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.backends import compile_run
from spikeforge_targets.backends.norse_backend import NorseBackend

pytest.importorskip("nir")


class FakeLIF:
    """A stand-in Norse LIF with the reference recurrence."""

    def __init__(
        self,
        p: float,
        v_leak: float,
        v_threshold: float,
        v_reset: Optional[float] = None,
    ) -> None:
        """Store the leak fraction, leak, threshold, and reset."""
        self.p = p
        self.v_leak = v_leak
        self.v_threshold = v_threshold
        self.v_reset = v_reset

    def __call__(self, x: Any, state: Any = None) -> Tuple[Any, Any]:
        """Return the (spike, membrane) pair for one step."""
        prev = torch.zeros_like(x) if state is None else state
        mem = prev + self.p * (self.v_leak - prev) + x
        spike = (mem > self.v_threshold).to(x.dtype)
        reset = torch.as_tensor(self.v_reset or 0.0, dtype=x.dtype,
                                device=x.device)
        return spike, torch.where(spike > 0, reset, mem)


class FakeLI:
    """A stand-in Norse LI integrator returning its membrane."""

    def __init__(self, p: float, v_leak: float) -> None:
        """Store the leak fraction and leak potential."""
        self.p = p
        self.v_leak = v_leak

    def __call__(self, x: Any, state: Any = None) -> Tuple[Any, Any]:
        """Return the (membrane, state) pair for one step."""
        prev = torch.zeros_like(x) if state is None else state
        mem = prev + self.p * (self.v_leak - prev) + x
        return mem, mem


@pytest.fixture
def norse(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``norse`` module for the duration of a test."""
    module = types.ModuleType("norse")
    module.LIF = FakeLIF
    module.LI = FakeLI
    module.__version__ = "0.0.test"
    monkeypatch.setitem(sys.modules, "norse", module)
    return module


def _graph() -> Any:
    """Return a norse-representable zero-reset linear chain graph."""
    spec = chain([
        Stage("fc1", "linear", {"in_features": 8, "out_features": 5}),
        Stage("lif1", "leaky", {"beta": 0.9, "reset": "zero"}),
        Stage("fc2", "linear", {"in_features": 5, "out_features": 3}),
        Stage("lif2", "leaky", {"beta": 0.9, "reset": "zero"}),
    ])
    return to_nir(spec, build_module(spec))


def _spikes() -> torch.Tensor:
    """Return a fixed-seed spike train for the chain graph."""
    torch.manual_seed(0)
    return torch.rand(6, 2, 8)


def test_norse_absent_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the SDK the backend reports unavailable, never raises."""
    monkeypatch.delitem(sys.modules, "norse", raising=False)
    result = compile_run("norse", _graph(), _spikes())
    assert result.status == "unavailable"
    assert result.notes
    assert result.readout is None
    assert result.rewritten is not None


def test_norse_available_flag_follows_the_sdk(norse: types.ModuleType) -> None:
    """The backend advertises availability exactly when the SDK imports."""
    assert NorseBackend().available() is True


def test_norse_run_matches_the_reference(norse: types.ModuleType) -> None:
    """A stubbed Norse run compiles, executes, and compares cleanly."""
    result = compile_run("norse", _graph(), _spikes())
    assert result.status == "ok"
    assert result.ok() is True
    assert sorted(result.spikes) == ["lif1", "lif2"]
    assert result.compare is not None
    assert result.compare["readout"]["max_abs"] <= 1e-6
    assert result.rewritten is not None


def test_norse_refuses_a_graph_with_an_unsupported_node(
    norse: types.ModuleType,
) -> None:
    """A Delay node (subtract reset) is refused with a named reason."""
    graph = to_nir(presets.conv_net(
        in_channels=1, channels=2, num_classes=3, input_size=8
    ))
    result = compile_run("norse", graph, torch.rand(4, 2, 1, 8, 8))
    assert result.status == "error"
    assert any("Delay" in note for note in result.notes)
    assert result.compare is None
