"""Activation/membrane quantization inside the target drift check.

The fixtures are binary spike trains rather than continuous ``rand`` frames,
and the graphs carry weights scaled until the network actually fires: with
its freshly-initialised weights the small conv preset never spikes at any
stage on this input, so its readout is identically zero and a spike-based
drift figure could not observe anything. A dedicated test pins that
liveness so the fixture cannot quietly go dead again.
"""

import json
from typing import Any, Dict

import pytest
import torch

from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.ops_registry import (
    DELAY,
    INPUT,
    OUTPUT,
    SPIKING_KINDS,
)
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec
from spikeforge_targets.activation_quant import (
    NO_ACTIVATION_SCHEME,
    Calibration,
)
from spikeforge_targets.activation_quant_graph import (
    GraphActivationQuantizer,
    calibrate_graph,
)
from spikeforge_targets.activation_quant_keys import (
    CURRENT_SUFFIX,
    MEMBRANE_SUFFIX,
)
from spikeforge_targets.fixed_point import levels_for
from spikeforge_targets.quantize import (
    FIXTURE_SOURCE,
    NO_FIXTURE,
    NO_QUANTIZATION,
    quantize,
)
from spikeforge_targets.rewrite_drift import rewrite_drift

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)
_SYNAPTIC = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8,
    neuron="synaptic",
)
#: Node kinds whose output is never an activation record.
_EXACT = frozenset(SPIKING_KINDS) | {INPUT, OUTPUT, DELAY}
BOTH = "activation_membrane_int8"
#: Factor the seeded init weights are scaled by so the readout fires.
GAIN = 8.0


def _graph(spec: TopologySpec, gain: float = GAIN) -> Any:
    """Return ``spec`` exported with seeded weights scaled by ``gain``."""
    torch.manual_seed(0)
    module = build_module(spec)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.mul_(gain)
    return to_nir(spec, module)


def _spikes(steps: int = 8) -> torch.Tensor:
    """Return a fixed-seed binary spike train shaped for the conv presets."""
    torch.manual_seed(1)
    return (torch.rand(steps, 2, 1, 8, 8) < 0.3).float()


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


def test_cuba_state_is_keyed_as_current_and_membrane() -> None:
    """A CubaLIF node's tuple state lands under two named keys."""
    graph = _graph(_SYNAPTIC)
    cuba = [
        name for name, node in graph.nodes.items()
        if type(node).__name__ == "CubaLIF"
    ]
    assert cuba
    section = quantize(
        graph, "reference", _spikes(), activation="membrane_int8"
    ).report.to_dict()["activation"]
    names = {item["name"] for item in section["layers"]}
    for name in cuba:
        assert name + CURRENT_SUFFIX in names
        assert name + MEMBRANE_SUFFIX in names


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


def _membrane_drift(steps: int, scheme: str = "membrane_int8") -> float:
    """Return the reported membrane drift of a ``steps``-long run."""
    graph = _graph(_CONV)
    report = quantize(
        graph, "reference", _spikes(steps), activation=scheme
    ).report.to_dict()
    return float(report["drift"]["membranes"]["max_abs"])


def test_membrane_drift_is_reported_from_the_very_first_step() -> None:
    """A one-step run reports the snap it applied, not zero.

    The recorded membrane is the value the hook returned, so a single step
    already shows its own rounding. Recording the pre-hook value instead
    made this exactly zero while every membrane had in fact been
    quantized, and hid the current step's snap at every length -- a
    reported number that did not measure what it named.
    """
    assert _membrane_drift(1) > 0.0
    assert _membrane_drift(2) > 0.0
    assert _membrane_drift(8) > 0.0


def test_recorded_membranes_lie_on_the_grid_they_were_snapped_to() -> None:
    """Every recorded membrane value is a whole number of grid steps."""
    graph = _graph(_CONV)
    spikes = _spikes()
    calibration = calibrate_graph(graph, spikes, target="membrane")
    hook = GraphActivationQuantizer(
        "membrane_int8", calibration=calibration
    )
    result = NirInterpreter(graph, post_node=hook).run(spikes)
    assert result.membranes
    for name, trace in result.membranes.items():
        bound = calibration.bound(name + MEMBRANE_SUFFIX)
        assert bound is not None
        scale = bound / levels_for(8)
        codes = trace / scale
        assert torch.allclose(codes, torch.round(codes), atol=1e-4)


def test_snapping_the_recorded_membrane_changes_no_dynamics() -> None:
    """The trace snap is record-only; the carried snap is what propagates.

    Both runs snap the carried state identically, so their spikes must
    match exactly: nothing downstream reads the recorded membrane. The
    carried snap *does* move spikes against an unquantized run, which is
    the point of simulating it and is asserted separately.
    """
    graph = _graph(_CONV)
    spikes = _spikes()
    both = GraphActivationQuantizer("membrane_int8")

    def carried_only(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Any:
        new_output, new_state, _snapped = both(
            name, kind, output, state, membrane
        )
        return new_output, new_state, membrane

    traced = NirInterpreter(
        graph, post_node=GraphActivationQuantizer("membrane_int8")
    ).run(spikes)
    untraced = NirInterpreter(graph, post_node=carried_only).run(spikes)
    for name, train in untraced.spikes.items():
        assert torch.equal(train, traced.spikes[name])
        assert set(torch.unique(train).tolist()) <= {0.0, 1.0}
    assert not torch.equal(
        untraced.membranes["lif1__mem"], traced.membranes["lif1__mem"]
    )


def test_the_carried_snap_does_move_the_spike_trains() -> None:
    """Quantizing what is carried is not cosmetic: it flips real spikes."""
    graph = _graph(_CONV)
    spikes = _spikes()
    plain = NirInterpreter(graph).run(spikes)
    hooked = NirInterpreter(
        graph, post_node=GraphActivationQuantizer("membrane_int8")
    ).run(spikes)
    assert any(
        not torch.equal(train, hooked.spikes[name])
        for name, train in plain.spikes.items()
    )
