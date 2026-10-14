"""Membrane quantization drift, recorded and snapped per step."""

from typing import Any

import pytest
import torch
from _quant_helpers import _CONV, _graph, _spikes

from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.topology import presets
from spikeforge_targets.activation_quant_graph import (
    GraphActivationQuantizer,
    calibrate_graph,
)
from spikeforge_targets.activation_quant_keys import (
    CURRENT_SUFFIX,
    MEMBRANE_SUFFIX,
)
from spikeforge_targets.fixed_point import levels_for
from spikeforge_targets.quantize import quantize

pytest.importorskip("nir")

_SYNAPTIC = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8,
    neuron="synaptic",
)


def _membrane_drift(steps: int, scheme: str = "membrane_int8") -> float:
    """Return the reported membrane drift of a ``steps``-long run."""
    graph = _graph(_CONV)
    report = quantize(
        graph, "reference", _spikes(steps), activation=scheme
    ).report.to_dict()
    return float(report["drift"]["membranes"]["max_abs"])


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
