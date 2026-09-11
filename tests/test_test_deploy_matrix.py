"""Simulator-backed test-deploy matrix: honest cells, parity, and the CLI."""

import json
import sys
import types
from typing import Any, Optional, Tuple

import pytest
import torch

from spikeforge.cli import verify
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain
from spikeforge.topology.stage import Stage
from spikeforge_targets import probe, registry
from spikeforge_targets.backends import api
from spikeforge_targets.cli import target_cli
from spikeforge_targets.test_deploy import run_matrix

pytest.importorskip("nir")

#: Targets backed by an optional simulator/vendor SDK extra.
SDK_BACKED = ("norse", "lava_loihi2", "speck", "xylo", "spinnaker2")
#: Every registered target now has exactly one executable backend wired.
WIRED = ("reference", "norse", "lava_loihi2", "speck", "xylo", "spinnaker2")


def _graph() -> Any:
    """Return a linear chain both the Lava and Norse lowerings can express."""
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


@pytest.fixture
def no_sdks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force every optional SDK to report absent, on any machine."""
    monkeypatch.setattr(probe, "module_available", lambda name: False)
    monkeypatch.setattr(api, "module_available", lambda name: False)


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
        reset = torch.as_tensor(
            self.v_reset or 0.0, dtype=x.dtype, device=x.device
        )
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
def norse_sdk(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Install a fake ``norse`` module so the simulator reports available."""
    module = types.ModuleType("norse")
    module.LIF = FakeLIF
    module.LI = FakeLI
    module.__version__ = "0.0.test"
    monkeypatch.setitem(sys.modules, "norse", module)
    return module


def test_matrix_has_one_cell_per_registered_target() -> None:
    """Every registered target contributes exactly one cell, none dropped."""
    matrix = run_matrix(_graph(), _spikes())
    assert [cell.target for cell in matrix.cells] == registry.target_names()
    assert len(matrix.by_target()) == len(registry.target_names())
    assert all(cell.capability is not None for cell in matrix.cells)


def test_reference_cell_is_available_and_runs(no_sdks: None) -> None:
    """The reference cell is always available and compares cleanly."""
    cell = run_matrix(_graph(), _spikes()).by_target()["reference"]
    assert cell.available is True
    assert cell.backend is True
    assert cell.status == "ok"
    assert cell.parity is not None
    assert cell.parity_ok is True
    assert cell.parity["readout"]["max_abs"] == 0.0


def test_absent_sdk_is_unavailable_with_named_reason(no_sdks: None) -> None:
    """An SDK-backed simulator without its SDK is honest, not an error."""
    cells = run_matrix(_graph(), _spikes()).by_target()
    for name in SDK_BACKED:
        cell = cells[name]
        assert cell.available is False
        assert cell.backend is True
        assert cell.status == "unavailable"
        assert cell.reason
        assert cell.ok() is False
        assert cell.estimate is True
        assert cell.parity is None


def test_every_target_has_an_executable_backend() -> None:
    """Every registered target now has a wired backend, none declared-only."""
    cells = run_matrix(_graph(), _spikes()).by_target()
    assert set(cells) == set(WIRED)
    for name in registry.target_names():
        assert cells[name].backend is True


def test_unwired_target_is_named_not_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A target with no backend names that gap, not a failure or a skip."""
    monkeypatch.setattr(
        "spikeforge_targets.test_deploy.backend_for", lambda name: None
    )
    cells = run_matrix(_graph(), _spikes()).by_target()
    cell = cells["speck"]
    assert cell.backend is False
    assert cell.status == "unavailable"
    assert cell.status != "error"
    assert cell.reason and "speck" in cell.reason


def test_absent_sdks_do_not_fail_the_matrix(no_sdks: None) -> None:
    """Missing optional SDKs never flip the matrix verdict to failure."""
    matrix = run_matrix(_graph(), _spikes())
    assert matrix.ok() is True
    assert matrix.parity_targets() == ["reference"]
    assert matrix.reference_cell() is not None


def test_available_simulator_reports_parity(
    norse_sdk: types.ModuleType,
) -> None:
    """A present simulator runs and carries a within-tolerance parity cell."""
    matrix = run_matrix(_graph(), _spikes())
    cell = matrix.by_target()["norse"]
    assert cell.available is True
    assert cell.backend is True
    assert cell.status == "ok"
    assert cell.parity is not None
    assert cell.parity_ok is True
    assert "norse" in matrix.parity_targets()
    assert matrix.ok() is True


def test_matrix_is_deterministic_and_json_serialisable() -> None:
    """The same input yields the same JSON-serialisable matrix."""
    first = run_matrix(_graph(), _spikes()).to_dict()
    second = run_matrix(_graph(), _spikes()).to_dict()
    assert first == second
    assert json.dumps(first)
    assert first["estimate"] is True
    assert first["reference"] == "reference"
    assert first["steps"] == 6


def test_target_filter_limits_the_cells() -> None:
    """A target filter runs only the named cells, in order."""
    matrix = run_matrix(_graph(), _spikes(), targets=["reference", "norse"])
    assert [cell.target for cell in matrix.cells] == ["reference", "norse"]


def test_cli_test_deploy_runs_end_to_end(
    no_sdks: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """``test-deploy`` prints the whole matrix JSON and exits zero."""
    assert verify.main(["test-deploy", "--topology", "fc_legacy"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["estimate"] is True
    assert [cell["target"] for cell in payload["cells"]] == (
        registry.target_names()
    )
    reference = next(
        cell for cell in payload["cells"] if cell["target"] == "reference"
    )
    assert reference["status"] == "ok"
    assert reference["available"] is True


def test_cli_test_deploy_target_filter(
    no_sdks: None, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--targets`` restricts the matrix to the requested targets."""
    args = ["test-deploy", "--topology", "fc_legacy", "--targets", "reference"]
    assert verify.main(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert [cell["target"] for cell in payload["cells"]] == ["reference"]


def test_cli_report_helper_and_exit(no_sdks: None) -> None:
    """The report helper serialises and maps its verdict to a status."""
    report = target_cli.test_deploy_report("fc_legacy")
    assert json.dumps(report)
    assert report["ok"] is True
    assert target_cli.test_deploy_exit(report) == 0
