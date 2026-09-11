"""Deployment report shape, JSON safety, and deployability."""

import json

import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge_targets import probe
from spikeforge_targets.report import deployment_report

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)


def _spikes() -> torch.Tensor:
    """Return a fixed-seed spike train for the small conv preset."""
    torch.manual_seed(0)
    return torch.rand(4, 2, 1, 8, 8)


def test_report_is_json_serialisable_with_constraints() -> None:
    """The report serialises and carries the declared constraints."""
    report = deployment_report(_CONV, "reference")
    assert json.dumps(report)
    assert report["available"] is True
    assert report["deployable"] is True
    assert report["constraints"]["dtype"] == "float32"
    assert report["target"]["constraints"] == report["constraints"]
    assert report["nodes"]["counts"]["total"] > 0
    assert report["validation"] is None


def test_report_for_unavailable_target_is_produced() -> None:
    """An unavailable target still yields a marked, non-raising report."""
    report = deployment_report(_CONV, "norse")
    assert json.dumps(report)
    assert report["available"] is False
    assert report["deployable"] is False
    assert report["notes"]


def test_report_marks_unsupported_nodes_undeployable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsupported nodes make a target undeployable even when available."""
    monkeypatch.setattr(probe, "extra_available", lambda extra: True)
    report = deployment_report(_CONV, "norse")
    assert report["available"] is True
    assert report["nodes"]["unsupported"]
    assert report["deployable"] is False


def test_report_carries_validation_when_inputs_given() -> None:
    """Supplying a module and spikes attaches the validation report."""
    module = build_module(_CONV)
    report = deployment_report(
        _CONV, "reference", module=module, spikes=_spikes()
    )
    assert json.dumps(report)
    assert report["validation"] is not None
    assert report["validation"]["within_tolerance"] is True
    assert report["nodes"]["unsupported"] == []


def test_report_accepts_an_exported_graph() -> None:
    """A pre-exported graph classifies without a validation section."""
    report = deployment_report(to_nir(_CONV), "reference")
    assert json.dumps(report)
    assert report["validation"] is None
    assert report["nodes"]["counts"]["total"] > 0
