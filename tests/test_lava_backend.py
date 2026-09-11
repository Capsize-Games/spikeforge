"""Lava hardware path: honest absence, named paths, and a stubbed run."""

import sys
import types
from typing import Any, Dict

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import api as nir_api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.backends import api, compile_run
from spikeforge_targets.backends.lava_backend import LavaBackend

pytest.importorskip("nir")


@pytest.fixture
def lava(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``lava`` module for the duration of a test."""
    module = types.ModuleType("lava")
    module.__version__ = "0.0.test"
    monkeypatch.setitem(sys.modules, "lava", module)
    return module


def _graph() -> Any:
    """Return a lava-representable dense linear/neuron chain graph."""
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


def _stub_run(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    """Replace the isolated Lava touch-point with a deterministic stub."""

    def run(program: Any, spikes: Any, device: bool) -> Dict[str, Any]:
        """Return a zero readout of the reference readout's shape."""
        return {"readout": np.zeros((2, 3), np.float32), "path": path}

    monkeypatch.setattr(api, "lava_run", run)


def test_lava_absent_reports_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the SDK the hardware path reports unavailable, no raise."""
    monkeypatch.delitem(sys.modules, "lava", raising=False)
    result = compile_run("lava_loihi2", _graph(), _spikes())
    assert result.status == "unavailable"
    assert result.notes
    assert result.path is None


def test_lava_emulator_path_is_named(
    lava: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present SDK runs on the emulator and names that path."""
    _stub_run(monkeypatch, "loihi2_emulator")
    monkeypatch.delenv(api.LAVA_DEVICE_ENV, raising=False)
    result = compile_run("lava_loihi2", _graph(), _spikes())
    assert result.status == "ok"
    assert result.path == "loihi2_emulator"
    assert any("emulator" in note for note in result.notes)
    assert result.compare is not None


def test_lava_device_path_is_named(
    lava: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An opted-in device run names the device path, not the emulator."""
    _stub_run(monkeypatch, "loihi2_device")
    monkeypatch.setenv(api.LAVA_DEVICE_ENV, "1")
    result = compile_run("lava_loihi2", _graph(), _spikes())
    assert result.status == "ok"
    assert result.path == "loihi2_device"
    assert any("device" in note for note in result.notes)


def test_lava_legacy_device_env_still_opts_in(
    lava: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legacy SNN_LAVA_DEVICE name still opts into the device path."""
    _stub_run(monkeypatch, "loihi2_device")
    monkeypatch.delenv(api.LAVA_DEVICE_ENV, raising=False)
    monkeypatch.setenv(api.LAVA_DEVICE_LEGACY_ENV, "1")
    result = compile_run("lava_loihi2", _graph(), _spikes())
    assert result.status == "ok"
    assert result.path == "loihi2_device"


def _conv_graph() -> Any:
    """Return a linear ``Input -> Conv2d -> Output`` graph."""
    source = nir_api.node_class("Input")({"input": None})
    sink = nir_api.node_class("Output")({"output": None})
    weight = np.zeros((1, 1, 3, 3), np.float32)
    conv = nir_api.node_class("Conv2d")(
        None, weight, 1, 0, 1, 1, np.zeros(1, np.float32)
    )
    return nir_api.node_class("NIRGraph")(
        {"input": source, "conv": conv, "output": sink},
        [("input", "conv"), ("conv", "output")],
        type_check=False,
    )


def test_lava_lowering_refuses_conv_nodes(
    lava: types.ModuleType,
) -> None:
    """A convolution node is refused with its node and kind named."""
    with pytest.raises(ValueError) as excinfo:
        LavaBackend().compile(_conv_graph(), None)
    assert "Conv2d" in str(excinfo.value)


def test_lava_lowering_is_json_free_program(
    lava: types.ModuleType,
) -> None:
    """The lowered program lists the chain's linear and neuron layers."""
    program = LavaBackend().compile(_graph(), None)
    kinds = [layer["kind"] for layer in program["layers"]]
    assert kinds == ["Linear", "LIF", "Linear", "LIF"]
    assert program["layers"][0]["size"] == 5
