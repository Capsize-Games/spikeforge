"""Checkpoint topology persistence and the legacy fallback."""

from typing import Any

import pytest
import torch

from snn_interpreter.network import model_store
from snn_interpreter.network.spiking_net import SpikingNet
from snn_interpreter.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _engine(**kwargs: Any) -> TrainingEngine:
    """Return a CPU engine with small, download-free defaults."""
    defaults: dict = {"dataset": "mnist", "num_steps": 4, "device": "cpu"}
    defaults.update(kwargs)
    return TrainingEngine(**defaults)


def test_conv_checkpoint_rebuilds_topology() -> None:
    """A saved non-default topology is restored from its checkpoint meta."""
    torch.manual_seed(0)
    engine = _engine(topology="conv_net", topology_params={"channels": 2})
    engine.save("conv_round_trip")
    restored = _engine(checkpoint="conv_round_trip")
    assert restored.topology == "conv_net"
    for key, value in restored.net.state_dict().items():
        assert torch.equal(value, engine.net.state_dict()[key])
    metrics = restored._train_batch(
        torch.rand(2, 1, 28, 28), torch.randint(0, 10, (2,))
    )
    assert metrics["loss"] > 0.0


def test_legacy_checkpoint_defaults_to_fc_legacy() -> None:
    """A checkpoint without a topology field still loads as fc_legacy."""
    torch.manual_seed(0)
    net = SpikingNet(hidden=8, beta=0.5, num_classes=10)
    meta = {"dataset": "mnist", "hidden": 8, "beta": 0.5,
            "num_classes": 10}
    model_store.save("legacy_ckpt", net, meta)
    restored = _engine(hidden=8, beta=0.5, checkpoint="legacy_ckpt")
    assert restored.topology == "fc_legacy"
    assert isinstance(restored.net, SpikingNet)
    assert torch.equal(
        restored.net.state_dict()["_fc1.weight"],
        net.state_dict()["_fc1.weight"],
    )


def test_describe_surfaces_topology() -> None:
    """The model listing labels a checkpoint with its topology."""
    torch.manual_seed(0)
    engine = _engine(topology="conv_net", topology_params={"channels": 2})
    engine.save("conv_listed")
    listed = {item["name"]: item for item in model_store.list_models()}
    assert listed["conv_listed"]["topology"] == "conv_net"
