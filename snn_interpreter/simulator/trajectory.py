"""The timed result of a single simulator run."""

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class Trajectory:
    """Timed outputs and, when recorded, per-stage traces of a run.

    ``logits`` is the readout averaged over ``steps``. ``spikes`` and
    ``membranes`` map neuron stage names to ``[T, B, N]`` (or ``[T, B, C, H,
    W]`` for spatial stages) trace tensors; they are empty when the run did
    not record that quantity.
    """

    steps: int
    logits: torch.Tensor
    spikes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    membranes: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def recorded(self) -> bool:
        """Return True when any per-step trace was captured."""
        return bool(self.spikes or self.membranes)
