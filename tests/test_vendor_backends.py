"""Vendor simulator backends: probes, honest absence, and stubbed runs.

Speck (Sinabs), Xylo (Rockpool), and SpiNNaker2 follow the same backend
protocol as Norse and Lava. These tests pin that contract without any real
SDK: absence is forced through ``sys.modules``, a minimal stubbed SDK proves
the capability check, and the isolated run touch-point is replaced so a
completed run can be checked for its ``path``, ``estimate``, and parity.
"""

import json
import sys
import types
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets.backends import BACKENDS, api, backend_for, compile_run
from spikeforge_targets.test_deploy import run_matrix

pytest.importorskip("nir")

#: Vendor target -> (SDK module, a capability attribute, run touch-point,
#: emitted simulator path).
VENDORS: Dict[str, Tuple[str, str, str, str]] = {
    "speck": ("sinabs", "Network", "sinabs_run", "speck_simulator"),
    "xylo": ("rockpool", "XyloSim", "rockpool_run", "xylo_simulator"),
    "spinnaker2": (
        "spinnaker2",
        "Spinnaker2",
        "spinnaker2_run",
        "spinnaker2_simulator",
    ),
}


def _graph() -> Any:
    """Return a bias-free linear chain all three vendor targets support.

    A ``bias: False`` linear stage renders a NIR ``Linear`` node rather than an
    ``Affine`` one, so the graph is genuinely target-ready for Speck (which
    declares ``Linear`` but not ``Affine``) as well as Xylo and SpiNNaker2.
    """
    spec = chain([
        Stage("fc1", "linear",
              {"in_features": 8, "out_features": 5, "bias": False}),
        Stage("lif1", "leaky", {"beta": 0.9, "reset": "zero"}),
        Stage("fc2", "linear",
              {"in_features": 5, "out_features": 3, "bias": False}),
        Stage("lif2", "leaky", {"beta": 0.9, "reset": "zero"}),
    ])
    return to_nir(spec, build_module(spec))


def _spikes() -> torch.Tensor:
    """Return a fixed-seed spike train for the chain graph."""
    torch.manual_seed(0)
    return torch.rand(6, 2, 8)


def _absent(monkeypatch: pytest.MonkeyPatch, module: str) -> None:
    """Force ``module`` to look uninstalled, on any machine."""
    monkeypatch.setitem(sys.modules, module, None)


def _present(
    monkeypatch: pytest.MonkeyPatch, module: str, capability: str
) -> types.ModuleType:
    """Install a minimal fake SDK exposing exactly one capability symbol."""
    fake = types.ModuleType(module)
    fake.__version__ = "0.0.test"
    setattr(fake, capability, type("Capability", (), {}))
    monkeypatch.setitem(sys.modules, module, fake)
    return fake


def _stub_run(
    monkeypatch: pytest.MonkeyPatch, run_attr: str, path: str
) -> None:
    """Replace a vendor SDK touch-point with a deterministic zero readout."""

    def run(program: Any, spikes: Any) -> Dict[str, Any]:
        """Return a zero readout of the reference readout's shape."""
        return {"readout": np.zeros((2, 3), np.float32), "path": path}

    monkeypatch.setattr(api, run_attr, run)


def test_vendor_backends_are_registered() -> None:
    """Every vendor target has exactly one executable backend wired."""
    for name in VENDORS:
        assert name in BACKENDS
        assert backend_for(name) is not None


def test_vendor_absent_is_unavailable_with_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without its SDK a vendor backend is honest, never a raised import."""
    for name, (module, _cap, _run, _path) in VENDORS.items():
        _absent(monkeypatch, module)
        assert backend_for(name).available() is False
        result = compile_run(name, _graph(), _spikes())
        assert result.status == "unavailable"
        assert result.path is None
        assert any(module in note or name in note for note in result.notes)
        assert result.to_dict()["estimate"] is True


def test_vendor_capability_check_rejects_a_bare_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed module with no simulator capability is not trusted."""
    for name, (module, _cap, _run, _path) in VENDORS.items():
        bare = types.ModuleType(module)
        monkeypatch.setitem(sys.modules, module, bare)
        assert api.backend_available(name) is False
        result = compile_run(name, _graph(), _spikes())
        assert result.status == "unavailable"


def test_vendor_capability_check_accepts_the_symbol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installed module exposing the capability symbol is trusted."""
    for name, (module, cap, _run, _path) in VENDORS.items():
        _present(monkeypatch, module, cap)
        assert api.backend_available(name) is True


@pytest.mark.parametrize("name", sorted(VENDORS))
def test_vendor_stubbed_run_is_ok_and_named(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A present SDK runs, names its simulator path, and stays an estimate."""
    module, cap, run_attr, path = VENDORS[name]
    _present(monkeypatch, module, cap)
    _stub_run(monkeypatch, run_attr, path)
    result = compile_run(name, _graph(), _spikes())
    assert result.status == "ok"
    assert result.ok() is True
    assert result.path == path
    assert result.estimate is True
    assert result.compare is not None
    assert result.rewritten is not None
    assert any("simulator" in note for note in result.notes)
    assert result.to_dict()["estimate"] is True


@pytest.mark.parametrize("name", sorted(VENDORS))
def test_vendor_matrix_cell_is_available(
    name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The matrix reports a present vendor as available and estimated."""
    module, cap, run_attr, path = VENDORS[name]
    _present(monkeypatch, module, cap)
    _stub_run(monkeypatch, run_attr, path)
    cell = run_matrix(_graph(), _spikes()).by_target()[name]
    assert cell.available is True
    assert cell.backend is True
    assert cell.status == "ok"
    assert cell.path == path
    assert cell.estimate is True
    assert cell.parity is not None


def test_vendor_matrix_reports_absent_without_failing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absent vendor SDKs make the matrix honest and still green."""
    for module, _cap, _run, _path in VENDORS.values():
        _absent(monkeypatch, module)
    matrix = run_matrix(_graph(), _spikes())
    assert matrix.ok() is True
    for name in VENDORS:
        cell = matrix.by_target()[name]
        assert cell.available is False
        assert cell.backend is True
        assert cell.status == "unavailable"
        assert cell.reason
        assert cell.estimate is True


def test_vendor_matrix_json_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The same vendor stubs yield the same JSON-serialisable matrix."""
    for module, cap, run_attr, path in VENDORS.values():
        _present(monkeypatch, module, cap)
        _stub_run(monkeypatch, run_attr, path)
    first = run_matrix(_graph(), _spikes()).to_dict()
    second = run_matrix(_graph(), _spikes()).to_dict()
    assert first == second
    assert json.dumps(first)
    assert first["estimate"] is True
    for cell in first["cells"]:
        assert cell["estimate"] is True
