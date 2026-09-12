"""One timestep's decision produced by a stateful inference session."""

from dataclasses import dataclass
from typing import Mapping

import torch


@dataclass(frozen=True)
class Prediction:
    """One timestep's readout and the session's cumulative class evidence.

    ``logits`` is the readout for this single step, ``class_totals`` is the
    readout summed over every step since the last reset, and ``steps`` is how
    many steps have been taken. Dividing ``class_totals`` by ``steps`` gives
    the mean readout, which is exactly what the closed-loop
    :func:`~spikeforge.simulator.runner.run` reports as its ``logits``.
    ``label`` is the class index with the highest cumulative evidence.
    """

    logits: torch.Tensor
    spikes: Mapping[str, torch.Tensor]
    class_totals: torch.Tensor
    steps: int
    label: int

    @property
    def mean_logits(self) -> torch.Tensor:
        """Return the mean readout across every step since the reset."""
        return self.class_totals / max(1, int(self.steps))

    @property
    def predicted(self) -> int:
        """Return the class index with the highest cumulative evidence."""
        return int(self.mean_logits.argmax(dim=-1).reshape(-1)[0])
