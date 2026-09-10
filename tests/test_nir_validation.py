"""End-to-end drift validation of presets against their NIR graphs."""

import json
from typing import Dict, Tuple

import pytest
import torch

from snn_interpreter.nir_bridge import drift
from snn_interpreter.nir_bridge.exporter import to_nir
from snn_interpreter.nir_bridge.validator import validate
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module
from snn_interpreter.topology.spec import TopologySpec

pytest.importorskip("nir")

_Shapes = Tuple[int, ...]
_Cases = Tuple[TopologySpec, _Shapes]

_PRESETS = ("fc_legacy", "fc_small", "conv_net", "recurrent_net")
_CASES: Dict[str, _Cases] = {
    "fc_legacy": (
        presets.fc_legacy(hidden=8, beta=0.5, num_classes=3, input_size=5),
        (6, 2, 5),
    ),
    "fc_small": (
        presets.fc_small(hidden=8, beta=0.9, num_classes=3, input_size=12),
        (6, 2, 1, 3, 4),
    ),
    "conv_net": (
        presets.conv_net(
            in_channels=1, channels=2, num_classes=3, input_size=8
        ),
        (6, 2, 1, 8, 8),
    ),
    "recurrent_net": (
        presets.recurrent_net(
            hidden=6, beta=0.9, num_classes=3, input_size=4
        ),
        (6, 2, 4),
    ),
}


def _case(name: str) -> _Cases:
    """Return the preset ``name`` and the spike-train shape it consumes."""
    return _CASES[name]


def _run(name: str) -> Tuple[TopologySpec, object, torch.Tensor]:
    """Build a preset module and a fixed-seed spike train for it."""
    spec, shape = _case(name)
    torch.manual_seed(0)
    module = build_module(spec)
    spikes = torch.rand(*shape)
    return spec, module, spikes


@pytest.mark.parametrize("name", _PRESETS)
def test_preset_validates_within_tolerance(name: str) -> None:
    """Every shipped preset matches its exported graph on a fixed seed."""
    spec, module, spikes = _run(name)
    report = validate(spec, module, spikes)
    assert report["within_tolerance"] is True
    assert report["layers"]
    assert json.dumps(report)


@pytest.mark.parametrize("name", _PRESETS)
def test_report_is_json_serialisable(name: str) -> None:
    """The report survives ``json.dumps`` unchanged for streaming."""
    spec, module, spikes = _run(name)
    report = validate(spec, module, spikes)
    assert json.dumps(report)


def test_perturbed_weight_fails_and_names_layer() -> None:
    """A perturbed exported weight flips the verdict and names the layer."""
    spec, module, spikes = _run("fc_legacy")
    graph = to_nir(spec, module)
    graph.nodes["_fc1"].weight = graph.nodes["_fc1"].weight + 0.5
    report = validate(spec, module, spikes, graph=graph)
    assert report["within_tolerance"] is False
    assert report["worst"]["layer"] == "_lif1"


def test_drift_metrics_are_plain_floats() -> None:
    """Drift metrics are JSON-able numbers, never tensors."""
    actual = torch.tensor([0.0, 1.0, 0.0, 0.0])
    reference = torch.tensor([0.0, 1.0, 1.0, 0.0])
    metrics = drift.compare(actual, reference)
    assert all(isinstance(value, float) for value in metrics.values())
    assert metrics["max_abs"] == 1.0
    assert metrics["agreement"] == pytest.approx(0.75)
