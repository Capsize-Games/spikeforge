"""Seed helper determinism for topology initialisation."""

from typing import Any, Dict

import torch

from snn_interpreter.topology import registry
from snn_interpreter.tracking.seed import set_seed


def _weights(seed: int) -> Dict[str, Any]:
    """Build fc_small after seeding and return its state dict."""
    set_seed(seed)
    _, net = registry.build_topology("fc_small", {"hidden": 16})
    return net.state_dict()


def test_same_seed_rebuilds_identical_weights() -> None:
    """Two builds from the same seed agree on every parameter."""
    first = _weights(1234)
    second = _weights(1234)
    assert set(first) == set(second)
    for key, value in first.items():
        assert torch.equal(value, second[key])


def test_different_seeds_diverge() -> None:
    """A different seed changes the initial parameters."""
    first = _weights(1)
    second = _weights(2)
    assert any(
        not torch.equal(value, second[key])
        for key, value in first.items()
    )
