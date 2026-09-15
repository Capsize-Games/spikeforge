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


def test_report_names_declared_schemes_unapplied_without_a_module() -> None:
    """A spec-only report names both declared schemes without applying."""
    section = deployment_report(_CONV, "lava_loihi2")["quantization"]
    assert section["scheme"] == "weight_int8"
    assert section["applied"] is False
    activation = section["activation"]
    assert activation["scheme"] == "none"
    assert activation["applied"] is False
    assert activation["reason"]


def test_report_runs_a_requested_activation_scheme_with_inputs() -> None:
    """A built module plus spikes lets the requested scheme run and report."""
    report = deployment_report(
        _CONV,
        "reference",
        module=build_module(_CONV),
        spikes=_spikes(),
        activation="activation_membrane_int8",
    )
    section = report["quantization"]
    assert json.dumps(report)
    assert section["activation"]["applied"] is True
    assert section["activation"]["counts"]["steps"] == _spikes().size(0)
    assert section["drift"]["includes"] == ["activation", "membrane"]


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
