"""Quantized summaries: the C envelope with simulation-identical grids."""

import json
import math
from typing import Any, Dict

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge_targets.errors import QuantizationError
from spikeforge_targets.quantize import quantize
from spikeforge_targets.quantized_summary import export_quantized_summary

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)


def _graph() -> Any:
    """Return the small conv preset exported with seeded weights."""
    torch.manual_seed(0)
    return to_nir(_CONV, build_module(_CONV))


def _affine_graph(weight: np.ndarray) -> Any:
    """Return a small ``Input -> Linear -> Output`` graph of ``weight``."""
    source = api.node_class("Input")({"input": None})
    sink = api.node_class("Output")({"output": None})
    linear = api.node_class("Linear")(weight)
    return api.node_class("NIRGraph")(
        {"input": source, "fc": linear, "output": sink},
        [("input", "fc"), ("fc", "output")],
        type_check=False,
    )


def _decode(envelope: Dict[str, Any]) -> np.ndarray:
    """Return ``envelope`` decoded the way the C loader decodes it."""
    data = np.asarray(envelope["data"], dtype=np.float64)
    return (data - envelope["zero_point"]) * envelope["scale"]


def _weights_of(summary: Dict[str, Any], name: str) -> Any:
    """Return the ``weight`` param of node ``name`` in ``summary``."""
    for record in summary["nodes"]:
        if record["name"] == name:
            return record["params"]["weight"]
    raise AssertionError(f"node {name!r} is missing from the summary")


def test_none_matches_graph_summary_byte_for_byte() -> None:
    """The ``none`` scheme is the float summary, with no reformatting."""
    graph = _graph()
    for target in (_CONV, graph):
        expected = json.dumps(graph_summary(target), sort_keys=True)
        actual = json.dumps(
            export_quantized_summary(target, "none"), sort_keys=True
        )
        assert actual == expected


def test_int8_envelopes_decode_to_simulated_weights() -> None:
    """Every int8 envelope decodes to the drift simulation's weights."""
    graph = _graph()
    simulated = quantize(graph, "lava_loihi2").graph
    assert simulated is not graph
    summary = json.loads(
        json.dumps(export_quantized_summary(graph, "weight_int8"))
    )
    enveloped = set()
    for record in summary["nodes"]:
        weight = record["params"].get("weight")
        if not isinstance(weight, dict):
            continue
        enveloped.add(record["name"])
        assert set(weight) == {"data", "dtype", "scale", "zero_point"}
        assert weight["dtype"] == "int8"
        assert math.isfinite(weight["scale"]) and weight["scale"] > 0.0
        assert isinstance(weight["zero_point"], int)
        expected = np.asarray(
            simulated.nodes[record["name"]].weight, dtype=np.float64
        )
        assert _decode(weight).shape == expected.shape
        assert np.allclose(_decode(weight), expected, rtol=1e-6, atol=1e-7)
    assert enveloped == {
        layer["node"]
        for layer in quantize(graph, "lava_loihi2").report.layers
    }
    assert enveloped


def test_uint8_envelopes_decode_to_simulated_weights() -> None:
    """Every uint8 envelope decodes to the drift simulation's weights."""
    graph = _graph()
    simulated = quantize(graph, "xylo").graph
    summary = json.loads(
        json.dumps(export_quantized_summary(graph, "weight_uint8"))
    )
    enveloped = set()
    for record in summary["nodes"]:
        weight = record["params"].get("weight")
        if not isinstance(weight, dict):
            continue
        enveloped.add(record["name"])
        assert weight["dtype"] == "uint8"
        assert math.isfinite(weight["scale"]) and weight["scale"] > 0.0
        expected = np.asarray(
            simulated.nodes[record["name"]].weight, dtype=np.float64
        )
        assert np.allclose(_decode(weight), expected, rtol=1e-6, atol=1e-7)
    assert enveloped == {
        layer["node"]
        for layer in quantize(graph, "xylo").report.layers
    }
    assert enveloped


def test_envelope_data_preserves_weight_shapes() -> None:
    """Envelope data nests exactly like the plain weight it replaces."""
    graph = _graph()
    plain = graph_summary(graph)
    for scheme in ("weight_int8", "weight_uint8"):
        summary = export_quantized_summary(graph, scheme)
        for record in summary["nodes"]:
            weight = record["params"].get("weight")
            if not isinstance(weight, dict):
                continue
            expected = np.asarray(
                _weights_of(plain, record["name"]), dtype=np.float64
            )
            assert _decode(weight).shape == expected.shape
            flat = np.asarray(weight["data"]).ravel().tolist()
            assert flat and all(isinstance(code, int) for code in flat)


def test_biases_and_other_params_stay_plain() -> None:
    """Only ``weight`` is enveloped, matching the drift simulation."""
    graph = _graph()
    plain = graph_summary(graph)
    summary = export_quantized_summary(graph, "weight_int8")
    seen_bias = False
    for record in summary["nodes"]:
        params = record["params"]
        for name, value in params.items():
            if name == "weight" and isinstance(value, dict):
                continue
            assert value == _plain_param(plain, record["name"], name)
        if isinstance(params.get("bias"), list):
            seen_bias = True
    assert seen_bias


def _plain_param(summary: Dict[str, Any], node: str, param: str) -> Any:
    """Return one plain-summary parameter for the untouched check."""
    for record in summary["nodes"]:
        if record["name"] == node:
            return record["params"][param]
    raise AssertionError(f"node {node!r} is missing from the summary")


@pytest.mark.parametrize(
    "weight", [np.zeros((2, 3), np.float32), np.full((2, 3), 2.5, np.float32)]
)
@pytest.mark.parametrize("scheme", ["weight_int8", "weight_uint8"])
def test_degenerate_envelopes_are_loadable_and_exact(
    weight: np.ndarray, scheme: str
) -> None:
    """Degenerate grids keep a positive scale and reproduce the values."""
    graph = _affine_graph(weight)
    summary = export_quantized_summary(graph, scheme)
    envelope = _weights_of(summary, "fc")
    assert math.isfinite(envelope["scale"]) and envelope["scale"] > 0.0
    assert np.allclose(_decode(envelope), weight, atol=1e-9)


def test_near_constant_envelope_stays_within_eps() -> None:
    """A sub-EPS range decodes within EPS of the simulated values."""
    weight = np.array([[1.0, 1.0 + 5e-13]], dtype=np.float32)
    summary = export_quantized_summary(
        _affine_graph(weight), "weight_uint8"
    )
    envelope = _weights_of(summary, "fc")
    assert math.isfinite(envelope["scale"]) and envelope["scale"] > 0.0
    assert float(np.abs(_decode(envelope) - weight).max()) <= 1e-12


def test_empty_weight_envelope_loads() -> None:
    """An empty weight carries empty data with a positive scale.

    Nested-list JSON cannot preserve empty dimensions (the float summary
    has the same property), so the envelope holds ``[]`` like a plain
    empty weight would.
    """
    summary = export_quantized_summary(
        _affine_graph(np.zeros((0, 3), np.float32)), "weight_uint8"
    )
    envelope = _weights_of(summary, "fc")
    assert math.isfinite(envelope["scale"]) and envelope["scale"] > 0.0
    assert envelope["data"] == []
    assert _decode(envelope).shape == (0,)


def test_spec_input_quantizes_placeholders_as_zeros() -> None:
    """A bare spec exports degenerate envelopes decoding to zeros."""
    for scheme in ("weight_int8", "weight_uint8"):
        summary = export_quantized_summary(_CONV, scheme)
        enveloped = 0
        for record in summary["nodes"]:
            weight = record["params"].get("weight")
            if not isinstance(weight, dict):
                continue
            enveloped += 1
            assert weight["scale"] > 0.0
            peak = float(np.abs(_decode(weight)).max(initial=0.0))
            assert peak <= 1e-12
        assert enveloped


def test_unknown_scheme_is_refused() -> None:
    """An unimplemented scheme is named, never guessed."""
    with pytest.raises(QuantizationError, match="weight_int4"):
        export_quantized_summary(_graph(), "weight_int4")


def test_non_finite_weights_are_refused() -> None:
    """Weights no finite grid can represent are refused per scheme."""
    weight = np.full((2, 2), float("inf"), dtype=np.float32)
    for scheme in ("weight_int8", "weight_uint8"):
        with pytest.raises(QuantizationError):
            export_quantized_summary(_affine_graph(weight), scheme)



