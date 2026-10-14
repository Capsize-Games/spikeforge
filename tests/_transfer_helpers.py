"""The checkpoint writer shared by the hub transfer tests."""

import torch

from spikeforge.training.training_engine import TrainingEngine


def _save_checkpoint(name: str = "transfer_ckpt") -> None:
    """Train one real step and save a preset-shaped fc_small checkpoint."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 32},
    )
    engine.save(name)
