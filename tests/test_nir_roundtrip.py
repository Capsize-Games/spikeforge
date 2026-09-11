"""Fidelity reports from the export/persist/reload round-trip helper."""

import json
from typing import Any, Dict, Tuple

import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.roundtrip import roundtrip
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec

pytest.importorskip("nir")

_CASES: Dict[str, Tuple[TopologySpec, Tuple[int, ...]]] = {
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
        presets.recurrent_net(hidden=6, beta=0.9, num_classes=3, input_size=4),
        (6, 2, 4),
    ),
}
_NAMES = sorted(_CASES)


def _module_and_spikes(
    name: str,
) -> Tuple[TopologySpec, Any, torch.Tensor]:
    """Return the preset, its built module, and a fixed-seed spike train."""
    spec, shape = _CASES[name]
    torch.manual_seed(0)
    return spec, build_module(spec), torch.rand(*shape)


@pytest.mark.parametrize("name", _NAMES)
def test_roundtrip_reports_identity(name: str) -> None:
    """The persisted artifact reproduces the export with zero drift."""
    spec, module, spikes = _module_and_spikes(name)
    report = roundtrip(spec, module, spikes)
    assert report["identical"] is True
    assert report["readout"]["max_abs"] == 0.0
    assert report["readout"]["agreement"] == 1.0
    assert report["steps"] == spikes.size(0)
    assert json.dumps(report)


@pytest.mark.parametrize("name", _NAMES)
def test_roundtrip_metrics_cover_every_trace(name: str) -> None:
    """Every spike and membrane trace reported has zero error."""
    spec, module, spikes = _module_and_spikes(name)
    report = roundtrip(spec, module, spikes)
    assert report["spikes"]
    assert report["membranes"]
    for metrics in report["spikes"].values():
        assert metrics["max_abs"] == 0.0
    for metrics in report["membranes"].values():
        assert metrics["max_abs"] == 0.0


def test_roundtrip_reports_drift_for_perturbed_parameter() -> None:
    """Persisting a perturbed graph is reported as drift against the export."""
    spec, module, spikes = _module_and_spikes("fc_legacy")
    perturbed = to_nir(spec, module)
    weight = perturbed.nodes["_fc1"].weight
    perturbed.nodes["_fc1"].weight = weight + 0.5
    report = roundtrip(spec, module, spikes, graph=perturbed)
    assert report["identical"] is False
    assert report["readout"]["max_abs"] > 0.0


def test_roundtrip_writes_the_requested_path(tmp_path: Any) -> None:
    """An explicit path is used and left on disk for later consumption."""
    spec, module, spikes = _module_and_spikes("fc_legacy")
    path = str(tmp_path / "artifact.nir.json")
    report = roundtrip(spec, module, spikes, path=path)
    assert report["path"] == path
    assert report["identical"] is True
    assert (tmp_path / "artifact.nir.json").exists()
