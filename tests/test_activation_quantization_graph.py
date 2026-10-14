"""Activation quantization inside the target drift check."""

import json
from typing import Any, Dict

import pytest
from _quant_helpers import _CONV, _graph, _spikes

from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.ops_registry import (
    DELAY,
    INPUT,
    OUTPUT,
    SPIKING_KINDS,
)
from spikeforge_targets.activation_quant import (
    NO_ACTIVATION_SCHEME,
    Calibration,
)
from spikeforge_targets.activation_quant_graph import (
    GraphActivationQuantizer,
    calibrate_graph,
)
from spikeforge_targets.quantize import (
    FIXTURE_SOURCE,
    NO_FIXTURE,
    NO_QUANTIZATION,
    quantize,
)
from spikeforge_targets.rewrite_drift import rewrite_drift

pytest.importorskip("nir")

#: Node kinds whose output is never an activation record.
_EXACT = frozenset(SPIKING_KINDS) | {INPUT, OUTPUT, DELAY}
BOTH = "activation_membrane_int8"

def _kinds(section: Dict[str, Any]) -> set:
    """Return the tensor kinds an activation section recorded."""
    return {item["kind"] for item in section["layers"]}


def test_fixture_readout_fires_and_the_unscaled_one_does_not() -> None:
    """The scaled fixture is live; the plain init would observe nothing."""
    live = NirInterpreter(_graph(_CONV)).run(_spikes())
    assert float(live.readout.sum()) > 0.0
    assert all(float(train.sum()) > 0.0 for train in live.spikes.values())
    dead = NirInterpreter(_graph(_CONV, gain=1.0)).run(_spikes())
    assert float(dead.readout.abs().sum()) == 0.0


def test_requested_scheme_runs_in_the_drift_check_and_is_named() -> None:
    """Weights plus a requested scheme report both, and the drift says so."""
    graph = _graph(_CONV)
    result = quantize(graph, "lava_loihi2", _spikes(), activation=BOTH)
    report = result.report.to_dict()
    assert json.dumps(report)
    assert report["applied"] is True
    section = report["activation"]
    assert section["applied"] is True
    assert section["scheme"] == BOTH
    assert section["target"] == "both"
    assert section["counts"]["steps"] == _spikes().size(0)
    assert _kinds(section) == {"activation", "membrane"}
    assert section["calibration"]["source"] == FIXTURE_SOURCE
    assert report["drift"]["includes"] == ["weights", "activation", "membrane"]


def test_rounding_alone_moves_the_reference_trajectory() -> None:
    """On the weight-free reference target every drift is the rounding's."""
    graph = _graph(_CONV)
    result = quantize(graph, "reference", _spikes(), activation=BOTH)
    report = result.report.to_dict()
    assert result.graph is graph
    assert report["applied"] is False
    assert report["reason"] == NO_QUANTIZATION
    assert report["activation"]["applied"] is True
    assert report["drift"]["includes"] == ["activation", "membrane"]
    assert report["drift"]["membranes"]["max_abs"] > 0.0
    assert report["drift"]["readout"]["max_abs"] > 0.0
    assert report["drift"]["spikes"]["agreement"] < 1.0
    layers = report["activation"]["layers"]
    assert any(item["max_abs"] > 0.0 for item in layers)


def test_declared_none_keeps_the_weight_only_drift_byte_for_byte() -> None:
    """Without a requested scheme the drift is the pre-existing weight one."""
    graph = _graph(_CONV)
    spikes = _spikes()
    result = quantize(graph, "lava_loihi2", spikes)
    report = result.report.to_dict()
    assert report["activation"]["applied"] is False
    assert report["activation"]["reason"] == NO_ACTIVATION_SCHEME
    assert report["drift"]["includes"] == ["weights"]
    plain = rewrite_drift(graph, result.graph, spikes)
    assert {**plain, "includes": ["weights"]} == report["drift"]


def test_unknown_scheme_is_refused_by_name_and_never_runs() -> None:
    """An unsupported scheme changes nothing but the refusal it records."""
    graph = _graph(_CONV)
    spikes = _spikes()
    weight_only = quantize(graph, "lava_loihi2", spikes).report.to_dict()
    refused = quantize(
        graph, "lava_loihi2", spikes, activation="activation_int4"
    ).report.to_dict()
    assert refused["activation"]["applied"] is False
    assert "activation_int4" in refused["activation"]["reason"]
    assert refused["drift"] == weight_only["drift"]


def test_scheme_without_a_fixture_is_reported_unapplied() -> None:
    """There is nothing to simulate on without spikes, and the report says."""
    report = quantize(_graph(_CONV), "lava_loihi2", activation=BOTH).report
    payload = report.to_dict()
    assert payload["applied"] is True
    assert payload["drift"] is None
    assert payload["activation"]["applied"] is False
    assert payload["activation"]["reason"] == NO_FIXTURE


def test_membrane_and_activation_schemes_snap_disjoint_kinds() -> None:
    """Each single-target scheme records only its own tensor kind."""
    graph = _graph(_CONV)
    membrane = quantize(
        graph, "reference", _spikes(), activation="membrane_int8"
    ).report.to_dict()
    assert _kinds(membrane["activation"]) == {"membrane"}
    assert membrane["drift"]["includes"] == ["membrane"]
    activation = quantize(
        graph, "reference", _spikes(), activation="activation_int8"
    ).report.to_dict()
    assert _kinds(activation["activation"]) == {"activation"}
    assert activation["drift"]["includes"] == ["activation"]


def test_spikes_and_virtual_nodes_are_never_listed() -> None:
    """Every activation record belongs to a computed, non-spiking node."""
    graph = _graph(_CONV)
    section = quantize(
        graph, "reference", _spikes(), activation=BOTH
    ).report.to_dict()["activation"]
    for item in section["layers"]:
        if item["kind"] != "activation":
            continue
        assert type(graph.nodes[item["name"]]).__name__ not in _EXACT


def test_calibration_keys_match_the_quantizer_and_are_recorded() -> None:
    """Every recorded tensor has a calibrated bound under the same key."""
    graph = _graph(_CONV)
    spikes = _spikes()
    calibration = calibrate_graph(graph, spikes, source="held-out")
    assert calibration.samples > 0
    section = quantize(
        graph, "reference", spikes, activation=BOTH, calibration=calibration
    ).report.to_dict()["activation"]
    assert section["calibration"]["source"] == "held-out"
    for item in section["layers"]:
        assert item["name"] in calibration.ranges


def test_a_tighter_calibration_clips_and_adds_error() -> None:
    """Bounds below the observed peak saturate, and the drift grows."""
    graph = _graph(_CONV)
    spikes = _spikes()
    fitted = calibrate_graph(graph, spikes)
    halved = Calibration(
        8,
        "both",
        {
            key: (low / 2, high / 2)
            for key, (low, high) in fitted.ranges.items()
        },
        fitted.samples,
        "halved",
    )
    loose = quantize(graph, "reference", spikes, activation=BOTH)
    tight = quantize(
        graph, "reference", spikes, activation=BOTH, calibration=halved
    )
    tight_layers = {
        item["name"]: item
        for item in tight.report.to_dict()["activation"]["layers"]
    }
    for name, item in tight_layers.items():
        bound = halved.bound(name)
        assert bound is not None
        assert max(abs(value) for value in item["after"]) <= bound + 1e-6
    loose_layers = {
        item["name"]: item
        for item in loose.report.to_dict()["activation"]["layers"]
    }
    # conv1 reads the unchanged input, so its snap error is pure grid error:
    # at most half a step when calibrated on the peak, half the peak when
    # the bound is halved.
    assert tight_layers["conv1"]["max_abs"] > loose_layers["conv1"]["max_abs"]
    tight_drift = tight.report.to_dict()["drift"]["membranes"]["max_abs"]
    loose_drift = loose.report.to_dict()["drift"]["membranes"]["max_abs"]
    assert tight_drift > loose_drift


def test_drift_hook_touches_only_the_rewritten_run() -> None:
    """The original graph runs plain; only the compared run is snapped."""
    graph = _graph(_CONV)
    spikes = _spikes()
    plain = rewrite_drift(graph, graph, spikes)
    assert plain["readout"]["max_abs"] == 0.0
    assert plain["membranes"]["max_abs"] == 0.0
    hooked = rewrite_drift(
        graph, graph, spikes, post_node=GraphActivationQuantizer(BOTH)
    )
    assert hooked["membranes"]["max_abs"] > 0.0
    assert hooked["membranes"]["nodes"] == plain["membranes"]["nodes"]


def test_hook_report_before_any_run_counts_nothing() -> None:
    """A fresh hook is applicable but has observed no steps or tensors."""
    hook = GraphActivationQuantizer(BOTH)
    assert hook.applied is True
    assert hook.targets == ("activation", "membrane")
    report = hook.report()
    assert report.applied is True
    assert report.counts() == {"layers": 0, "steps": 0}
    assert GraphActivationQuantizer("none").targets == ()
