"""Tests for additive per-neuron input-current capture."""

import torch

from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator.frames import normalise_frame
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module
from snn_interpreter.topology.spec import TopologySpec, chain
from snn_interpreter.topology.stage import Stage


def _chain() -> TopologySpec:
    """Return a linear-then-leaky chain used across these tests."""
    return chain(
        [
            Stage("fc", "linear", {"in_features": 3, "out_features": 4}),
            Stage("n", "leaky", {"beta": 0.5}),
        ]
    )


def test_current_capture_shape_matches_spikes() -> None:
    """Currents are recorded per neuron stage with the spike shape."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=5)
    )
    trajectory = run(module, torch.rand(4, 2, 5), current=True)
    assert set(trajectory.currents) == {"_lif1", "_lif2"}
    assert trajectory.currents["_lif1"].shape == (4, 2, 4)
    assert trajectory.currents["_lif2"].shape == (4, 2, 3)


def test_current_equals_neuron_input_activation() -> None:
    """The recorded current is the merged inbound activation of the neuron."""
    torch.manual_seed(0)
    module = build_module(_chain())
    spikes = torch.rand(3, 2, 3)
    trajectory = run(module, spikes, current=True)
    fc = module.get_submodule("fc")
    expected = torch.stack(
        [fc(normalise_frame(spikes[t], "linear")) for t in range(3)]
    )
    assert torch.allclose(trajectory.currents["n"], expected)


def test_educational_mode_enables_current_capture() -> None:
    """Educational mode records currents without the explicit flag."""
    module = build_module(_chain())
    trajectory = run(
        module, torch.rand(3, 2, 3), mode=ExecutionMode.EDUCATIONAL
    )
    assert "n" in trajectory.currents
    assert trajectory.recorded


def test_production_mode_with_flags_off_captures_nothing() -> None:
    """Production mode with every flag off records no trace at all."""
    module = build_module(_chain())
    trajectory = run(module, torch.rand(3, 2, 3))
    assert trajectory.currents == {}
    assert trajectory.spikes == {}
    assert trajectory.membranes == {}
    assert not trajectory.recorded


def test_current_capture_for_second_order_neuron() -> None:
    """Second-order neurons expose one current frame per step as well."""
    spec = chain(
        [
            Stage("fc", "linear", {"in_features": 3, "out_features": 4}),
            Stage("syn", "synaptic", {"alpha": 0.8, "beta": 0.9}),
        ]
    )
    trajectory = run(build_module(spec), torch.rand(3, 2, 3), current=True)
    assert trajectory.currents["syn"].shape == (3, 2, 4)
