"""Checkpoint and bundle writers shared by the bundle tests."""

from typing import Any

import torch

from spikeforge.serving.bundle import build
from spikeforge.training.training_engine import TrainingEngine

_NAME = "serving_ckpt"


def _checkpoint() -> str:
    """Save a small fc_small checkpoint and return its name."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)
    return _NAME


def _written(tmp_path: Any) -> str:
    """Build and write a bundle, returning its path."""
    out = str(tmp_path / "model.spkf")
    build(_checkpoint(), out=out)
    return out
