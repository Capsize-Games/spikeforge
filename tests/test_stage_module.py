"""Tests for single-step execution of a ``StageModule``."""

import torch
import torch.nn as nn

from spikeforge.topology import presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import chain, multi_branch, residual
from spikeforge.topology.stage import Stage
from spikeforge.topology.stage_module import PREV_KEY


def _identity_module(module: nn.Module, name: str) -> None:
    with torch.no_grad():
        linear = module.get_submodule(name)
        linear.weight.copy_(torch.eye(3))
        linear.bias.zero_()


def _linear(name: str) -> Stage:
    return Stage(name, "linear", {"in_features": 3, "out_features": 3})


def test_fc_step_shapes() -> None:
    """One step returns the documented shapes for a linear+LIF chain."""
    spec = chain(
        [
            Stage("fc", "linear", {"in_features": 4, "out_features": 6}),
            Stage("lif", "leaky", {"beta": 0.9}),
        ]
    )
    outputs, state = build_module(spec).step(torch.randn(3, 4))
    assert outputs["fc"].shape == (3, 6)
    assert outputs["lif"].shape == (3, 6)
    assert state["lif"][0].shape == (3, 6)


def test_state_keeps_previous_outputs() -> None:
    """The state carries the previous outputs for delayed edges."""
    module = build_module(
        presets.fc_legacy(hidden=4, beta=0.9, num_classes=2, input_size=3)
    )
    outputs, state = module.step(torch.randn(2, 3))
    assert set(state[PREV_KEY]) == set(outputs)


def test_state_threads_between_steps() -> None:
    """Membrane state evolves across successive steps."""
    spec = chain(
        [
            Stage("fc", "linear", {"in_features": 3, "out_features": 4}),
            Stage("lif", "leaky", {"beta": 0.9}),
        ]
    )
    module = build_module(spec)
    _, first = module.step(torch.ones(2, 3))
    _, second = module.step(torch.ones(2, 3), first)
    assert not torch.equal(first["lif"][0], second["lif"][0])


def test_residual_merge_sums_skip_and_branch() -> None:
    """A residual merge adds the identity skip to the branch output."""
    stages = [
        Stage("src", "add", {}),
        _linear("branch"),
        Stage("merge", "add", {}),
    ]
    spec = residual(stages, "src", "merge")
    module = build_module(spec)
    _identity_module(module, "branch")
    x = torch.randn(2, 3)
    outputs, _ = module.step(x)
    assert torch.allclose(outputs["merge"], 2 * x)


def test_multi_branch_merge_sums_branches() -> None:
    """A multi-branch merge adds both branch outputs."""
    spec = multi_branch(
        Stage("src", "add", {}),
        [[_linear("left")], [_linear("right")]],
        Stage("merge", "add", {}),
    )
    module = build_module(spec)
    _identity_module(module, "left")
    _identity_module(module, "right")
    x = torch.randn(2, 3)
    outputs, _ = module.step(x)
    assert torch.allclose(outputs["merge"], 2 * x)


def test_recurrent_threads_delayed_state() -> None:
    """The recurrent merge reads the previous step's feedback output."""
    spec = presets.recurrent_net(
        hidden=6, beta=0.9, num_classes=3, input_size=4
    )
    module = build_module(spec)
    first, state = module.step(torch.randn(2, 4))
    second, _ = module.step(torch.randn(2, 4), state)
    expected = second["lif1"] + first["rec"]
    assert torch.allclose(second["merge"], expected)
