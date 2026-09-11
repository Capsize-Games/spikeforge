"""Presets build correctly and match the legacy checkpoint keys."""

import pytest
import snntorch as snn
import torch

from spikeforge.network.spiking_net import SpikingNet
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module


def test_fc_legacy_state_dict_matches_spiking_net() -> None:
    """fc_legacy reproduces SpikingNet's exact state_dict keys."""
    hidden, beta, classes, size = 8, 0.5, 10, 784
    legacy = build_module(
        presets.fc_legacy(hidden, beta, classes, size)
    )
    net = SpikingNet(
        hidden=hidden, beta=beta, num_classes=classes, input_size=size
    )
    assert legacy.state_dict().keys() == net.state_dict().keys()


def test_fc_legacy_stage_names() -> None:
    """fc_legacy uses exactly the four legacy stage names."""
    spec = presets.fc_legacy(hidden=4, beta=0.5, num_classes=2, input_size=3)
    assert [stage.name for stage in spec.stages] == [
        "_fc1",
        "_lif1",
        "_fc2",
        "_lif2",
    ]
    spec.validate()


def test_fc_legacy_neuron_is_leaky() -> None:
    """The legacy neuron is snn.Leaky(beta=beta), matching SpikingNet."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.7, num_classes=2, input_size=3)
    )
    neuron = module.get_submodule("_lif1")
    assert isinstance(neuron, snn.Leaky)
    assert float(neuron.beta) == pytest.approx(0.7)


def test_fc_small_builds_and_steps() -> None:
    """fc_small builds and runs one step with the expected readout."""
    module = build_module(
        presets.fc_small(hidden=4, beta=0.9, num_classes=3, input_size=8)
    )
    outputs, _ = module.step(torch.randn(2, 8))
    assert outputs["lif2"].shape == (2, 3)


def test_conv_net_builds_and_steps() -> None:
    """conv_net builds and runs one step over an image batch."""
    module = build_module(
        presets.conv_net(
            in_channels=1, channels=4, num_classes=5, input_size=28
        )
    )
    outputs, _ = module.step(torch.randn(2, 1, 28, 28))
    assert outputs["out"].shape == (2, 5)


def test_conv_net_default_dropout_is_identity() -> None:
    """The default dropout probability is 0.0, changing nothing."""
    module = build_module(
        presets.conv_net(
            in_channels=1, channels=4, num_classes=5, input_size=28
        )
    )
    dropout = module.get_submodule("dropout")
    assert isinstance(dropout, torch.nn.Dropout)
    assert dropout.p == 0.0


def test_conv_net_dropout_param_sets_probability() -> None:
    """A non-zero ``dropout`` reaches the rendered nn.Dropout stage."""
    module = build_module(
        presets.conv_net(
            in_channels=1, channels=4, num_classes=5, input_size=28,
            dropout=0.3,
        )
    )
    dropout = module.get_submodule("dropout")
    assert dropout.p == pytest.approx(0.3)


def test_conv_net_with_dropout_still_builds_and_steps() -> None:
    """A dropout-enabled conv_net still runs one step end to end."""
    module = build_module(
        presets.conv_net(
            in_channels=1, channels=4, num_classes=5, input_size=28,
            dropout=0.5,
        )
    )
    outputs, _ = module.step(torch.randn(2, 1, 28, 28))
    assert outputs["out"].shape == (2, 5)


def test_recurrent_net_builds_and_validates() -> None:
    """recurrent_net is a valid graph and runs one step."""
    spec = presets.recurrent_net(
        hidden=6, beta=0.9, num_classes=3, input_size=4
    )
    spec.validate()
    outputs, _ = build_module(spec).step(torch.randn(2, 4))
    assert outputs["out"].shape == (2, 3)
