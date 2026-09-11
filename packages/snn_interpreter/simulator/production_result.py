"""The observable result of a production-mode run."""

from dataclasses import dataclass

import torch

from snn_interpreter.simulator.trajectory import Trajectory


@dataclass(frozen=True)
class ProductionResult:
    """A production run's trajectory plus how it was actually executed.

    ``compiled`` is True only when a compiled callable served the run;
    ``status`` carries the documented compile status (``"eager"``,
    ``"unavailable"``, ``"compiled"``, or ``"fallback"``) so a caller can tell
    a genuine compiled run from a transparent eager fallback.
    """

    trajectory: Trajectory
    compiled: bool
    status: str

    @property
    def logits(self) -> torch.Tensor:
        """Return the run's averaged readout logits."""
        return self.trajectory.logits
