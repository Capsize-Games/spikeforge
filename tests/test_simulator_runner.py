"""Tests for the generic simulator and its ``Trajectory`` contract."""

import torch

from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator.runner import run
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.edge import Edge
from spikeforge.topology.spec import TopologySpec, chain
from spikeforge.topology.stage import Stage


def test_run_conv_net_end_to_end() -> None:
    """conv_net runs over spatial frames and reads out ``[B, classes]``."""
    module = build_module(
        presets.conv_net(
            in_channels=1, channels=4, num_classes=5, input_size=28
        )
    )
    spikes = torch.rand(3, 2, 1, 28, 28)
    trajectory = run(module, spikes, track=True)
    assert trajectory.steps == 3
    assert trajectory.logits.shape == (2, 5)
    assert trajectory.spikes["lif1"].shape == (3, 2, 4, 28, 28)
    assert trajectory.spikes["out"].shape == (3, 2, 5)


def test_run_recurrent_net_end_to_end() -> None:
    """recurrent_net runs over feature frames and reads out ``[B, C]``."""
    module = build_module(
        presets.recurrent_net(
            hidden=6, beta=0.9, num_classes=3, input_size=4
        )
    )
    trajectory = run(module, torch.rand(4, 2, 4), track=True)
    assert trajectory.steps == 4
    assert trajectory.logits.shape == (2, 3)
    assert trajectory.spikes["lif1"].shape == (4, 2, 6)


def _undelayed(spec: TopologySpec) -> TopologySpec:
    """Return ``spec`` with every delayed edge treated as forward."""
    return TopologySpec(
        stages=spec.stages,
        edges=[Edge(edge.source, edge.target) for edge in spec.edges],
        input=spec.input,
        output=spec.output,
    )


def test_delayed_edge_threads_previous_step() -> None:
    """A delayed feedback edge changes state downstream at later steps."""
    spec = presets.recurrent_net(
        hidden=6, beta=0.9, num_classes=3, input_size=4
    )
    delayed = build_module(spec)
    undelayed = build_module(_undelayed(spec))
    undelayed.load_state_dict(delayed.state_dict())
    spikes = torch.rand(5, 2, 4)
    delayed_trace = run(delayed, spikes, membrane=True)
    undelayed_trace = run(undelayed, spikes, membrane=True)
    assert torch.allclose(
        delayed_trace.membranes["lif1"], undelayed_trace.membranes["lif1"]
    )
    assert not torch.allclose(
        delayed_trace.membranes["lif2"], undelayed_trace.membranes["lif2"]
    )


def test_leaky_membrane_traces_follow_neuron_layout() -> None:
    """Membrane capture works for the leaky neuron kind."""
    leaky = build_module(
        presets.fc_legacy(hidden=5, beta=0.5, num_classes=2, input_size=3)
    )
    trace = run(leaky, torch.rand(4, 2, 3), membrane=True)
    assert trace.membranes["_lif1"].shape == (4, 2, 5)


def test_synaptic_membrane_traces_follow_neuron_layout() -> None:
    """Membrane capture works for the synaptic neuron kind."""
    synaptic = build_module(
        chain(
            [
                Stage("fc", "linear", {"in_features": 3, "out_features": 5}),
                Stage("syn", "synaptic", {"alpha": 0.8, "beta": 0.9}),
            ]
        )
    )
    trace = run(synaptic, torch.rand(4, 2, 3), membrane=True)
    assert trace.membranes["syn"].shape == (4, 2, 5)


def test_recurrent_membrane_traces_follow_neuron_layout() -> None:
    """Membrane capture works for the recurrent neuron kind."""
    recurrent = build_module(
        chain(
            [
                Stage("fc", "linear", {"in_features": 3, "out_features": 4}),
                Stage(
                    "rl", "recurrent", {"beta": 0.9, "linear_features": 4}
                ),
            ]
        )
    )
    trace = run(recurrent, torch.rand(3, 2, 3), membrane=True)
    assert trace.membranes["rl"].shape == (3, 2, 4)


def test_production_mode_records_nothing() -> None:
    """Production mode with flags off records no traces, same logits."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=5)
    )
    spikes = torch.rand(4, 2, 5)
    quiet = run(module, spikes, mode=ExecutionMode.PRODUCTION)
    loud = run(module, spikes, track=True)
    assert quiet.spikes == {}
    assert quiet.membranes == {}
    assert not quiet.recorded
    assert torch.allclose(quiet.logits, loud.logits)


def test_module_moves_between_devices() -> None:
    """A built module moves via ``nn.Module.to`` without a name clash."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=2, input_size=3)
    )
    moved = module.to("cpu")
    param = next(moved.parameters())
    assert param.device.type == "cpu"


def test_educational_mode_records_full_trajectory() -> None:
    """Educational mode captures spikes and membranes without flags."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=5)
    )
    trajectory = run(
        module, torch.rand(3, 2, 5), mode=ExecutionMode.EDUCATIONAL
    )
    assert trajectory.recorded
    assert trajectory.spikes["_lif2"].shape == (3, 2, 3)
    assert trajectory.membranes["_lif2"].shape == (3, 2, 3)
