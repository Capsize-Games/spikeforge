"""Topology-general training engine behaviour (no dataset downloads)."""

from typing import Any

import pytest
import torch
from torch.nn.functional import cross_entropy

from spikeforge.network.spiking_net import SpikingNet
from spikeforge.simulator.runner import run
from spikeforge.training.training_engine import TrainingEngine

_LEGACY_KEYS = {"_fc1.weight", "_fc1.bias", "_fc2.weight", "_fc2.bias"}


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: dict = {"dataset": "mnist", "num_steps": 4, "device": "cpu"}
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_fc_legacy_builds_spiking_net() -> None:
    """The default topology still yields the legacy contract."""
    torch.manual_seed(0)
    engine = _engine(hidden=8, beta=0.5)
    assert isinstance(engine.net, SpikingNet)
    assert set(engine.net.state_dict()) >= _LEGACY_KEYS
    assert engine.topology == "fc_legacy"
    assert (engine.hidden, engine.beta) == (8, 0.5)


def test_fc_legacy_step_matches_simulator_reference() -> None:
    """One engine step reproduces the generic-simulator loss exactly."""
    torch.manual_seed(0)
    engine = _engine(hidden=8, beta=0.5)
    images = torch.rand(6, 1, 28, 28)
    targets = torch.randint(0, 10, (6,))
    with torch.no_grad():
        logits = run(engine.net, engine._encode_batch(images)).logits
        expected = cross_entropy(logits, targets).item()
    metrics = engine._train_batch(images, targets)
    assert metrics["loss"] == pytest.approx(expected, abs=1e-6)
    assert 0.0 <= metrics["train_accuracy"] <= 1.0


def test_feature_topology_keeps_flat_frames() -> None:
    """A feature topology receives flattened [T,B,784] frames."""
    engine = _engine(hidden=8)
    spikes = engine._encode_batch(torch.rand(3, 1, 28, 28))
    assert spikes.shape == (4, 3, 28 * 28)


def test_spatial_topology_reshapes_and_trains() -> None:
    """A conv topology gets [T,B,1,28,28] input and completes a step."""
    torch.manual_seed(0)
    engine = _engine(topology="conv_net", topology_params={"channels": 2})
    assert engine.topology == "conv_net"
    images = torch.rand(4, 1, 28, 28)
    assert engine._encode_batch(images).shape == (4, 4, 1, 28, 28)
    metrics = engine._train_batch(images, torch.randint(0, 10, (4,)))
    assert metrics["loss"] > 0.0
    assert 0.0 <= metrics["train_accuracy"] <= 1.0


def test_conv_topology_dropout_param_reaches_the_module() -> None:
    """``topology_params["dropout"]`` builds a matching nn.Dropout stage."""
    engine = _engine(
        topology="conv_net",
        topology_params={"channels": 2, "dropout": 0.4},
    )
    dropout = engine.net.get_submodule("dropout")
    assert dropout.p == pytest.approx(0.4)
