"""Target quantization: application, honest no-ops, and drift reporting."""

import json

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge_targets.quantize import NO_QUANTIZATION, quantize
from spikeforge_targets.registry import get_target
from spikeforge_targets.report import deployment_report
from spikeforge_targets.target_spec import TargetSpec

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)


def _affine_graph(weight: np.ndarray) -> object:
    """Return a small ``Input -> Linear -> Output`` graph of known weights."""
    source = api.node_class("Input")({"input": None})
    sink = api.node_class("Output")({"output": None})
    linear = api.node_class("Linear")(weight)
    return api.node_class("NIRGraph")(
        {"input": source, "fc": linear, "output": sink},
        [("input", "fc"), ("fc", "output")],
        type_check=False,
    )


def _spikes() -> torch.Tensor:
    """Return a fixed-seed spike train for the small conv preset."""
    torch.manual_seed(0)
    return torch.rand(4, 2, 1, 8, 8)


def _unknown_scheme_target() -> TargetSpec:
    """Return a target declaring a scheme this module cannot apply."""
    return TargetSpec(
        name="int4_chip",
        kind="hardware",
        description="placeholder with an unimplemented scheme",
        extra=None,
        supported=frozenset({"Input", "Output", "Linear"}),
        constraints={"quantization": "weight_int4"},
    )


def test_int8_scheme_clamps_weights_to_the_grid() -> None:
    """A weight_int8 target clamps weights onto the int8 levels."""
    weight = np.array([[-1.0, -0.5, 0.0, 0.5, 1.0]], np.float32)
    graph = _affine_graph(weight)
    result = quantize(graph, "lava_loihi2")
    assert result.applied() is True
    quantized = np.asarray(result.graph.nodes["fc"].weight)
    scale = 1.0 / 127.0
    expected = np.round(weight / scale) * scale
    assert np.allclose(quantized, expected, atol=1e-6)
    assert float(np.max(np.abs(quantized))) <= 1.0 + 1e-6
    assert np.asarray(graph.nodes["fc"].weight).tolist() == weight.tolist()


def test_uint8_scheme_is_asymmetric() -> None:
    """A weight_uint8 target shifts the range to 256 unsigned levels."""
    weight = np.array([[-1.0, 0.0, 1.0]], np.float32)
    result = quantize(_affine_graph(weight), "xylo")
    assert result.applied() is True
    assert result.report.scheme == "weight_uint8"
    quantized = np.asarray(result.graph.nodes["fc"].weight)
    assert float(quantized.min()) >= -1.0 - 1e-6
    assert float(quantized.max()) <= 1.0 + 1e-6


def test_none_target_reports_a_no_op() -> None:
    """A target declaring no quantization leaves the graph untouched."""
    graph = _affine_graph(np.ones((1, 2), np.float32))
    result = quantize(graph, "reference")
    assert result.applied() is False
    assert result.graph is graph
    assert result.report.reason == NO_QUANTIZATION
    assert result.report.counts() == {"layers": 0}


def test_unknown_scheme_is_reported_unapplied() -> None:
    """An unimplemented scheme is named, never guessed."""
    graph = _affine_graph(np.ones((1, 2), np.float32))
    result = quantize(graph, _unknown_scheme_target())
    assert result.applied() is False
    assert result.graph is graph
    assert "weight_int4" in result.report.reason


def test_conv_net_int8_reports_the_induced_drift() -> None:
    """Quantizing conv_net records per-layer ranges and measured drift."""
    graph = to_nir(_CONV, build_module(_CONV))
    result = quantize(graph, "lava_loihi2", _spikes())
    assert result.applied() is True
    assert result.report.counts()["layers"] > 0
    drift = result.report.drift
    assert drift is not None
    assert "within_tolerance" in drift
    assert drift["steps"] > 0


def test_report_is_json_serialisable_and_device_free() -> None:
    """The report serialises and never claims a device-quantized result."""
    report = quantize(_affine_graph(np.ones((1, 2), np.float32)), "speck")
    payload = report.report.to_dict()
    assert json.dumps(payload)
    assert "device" not in payload
    assert payload["scheme"] == "weight_int8"


def test_deployment_report_declares_unapplied_without_weights() -> None:
    """A spec-only report names the scheme but does not claim it applied."""
    report = deployment_report(_CONV, "lava_loihi2")
    section = report["quantization"]
    assert section["scheme"] == "weight_int8"
    assert section["applied"] is False
    assert "no built module" in section["reason"]


def test_deployment_report_applies_with_a_built_module() -> None:
    """Supplying a built module lets the report apply the quantization."""
    module = build_module(_CONV)
    report = deployment_report(_CONV, "lava_loihi2", module=module)
    assert report["quantization"]["applied"] is True
    assert report["quantization"]["counts"]["layers"] > 0


def test_registry_targets_all_declare_a_known_scheme() -> None:
    """Every shipped target's scheme is either known or honestly unknown."""
    from spikeforge_targets.quantize_schemes import SCHEMES

    for name in ("reference", "lava_loihi2", "xylo", "norse"):
        scheme = str(get_target(name).constraints.get("quantization"))
        assert scheme in SCHEMES
