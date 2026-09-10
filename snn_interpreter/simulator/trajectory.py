"""The timed result of a single simulator run."""

from dataclasses import dataclass, field
from typing import Mapping

import torch


@dataclass(frozen=True)
class Trajectory:
    """Timed outputs and, when recorded, per-stage traces of a run.

    ``logits`` is the readout averaged over ``steps``. ``spikes``,
    ``membranes`` and ``currents`` map neuron stage names to ``[T, B, N]``
    (or ``[T, B, C, H, W]`` for spatial stages) trace tensors; they are
    empty when the run did not record that quantity. ``currents`` is the
    merged inbound activation each neuron stage received at each step, i.e.
    its input current ``I[t]`` before the neuron update.
    """

    steps: int
    logits: torch.Tensor
    spikes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    membranes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    currents: Mapping[str, torch.Tensor] = field(default_factory=dict)

    @property
    def recorded(self) -> bool:
        """Return True when any per-step trace was captured."""
        return bool(self.spikes or self.membranes or self.currents)
