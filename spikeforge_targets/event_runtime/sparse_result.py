"""The result of a sparse/event-driven run: trajectory plus op counts."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping

from spikeforge.simulator.trajectory import Trajectory


@dataclass(frozen=True)
class SparseResult:
    """A sparse run's readout trajectory and its operation tally.

    ``trajectory`` follows the dense shape contract, so a sparse result is a
    drop-in for readout comparison. ``counts`` and ``per_stage`` hold the SOP,
    MAC, AC, and timestep tallies, and ``density`` is the input spike density.
    """

    trajectory: Trajectory
    counts: Mapping[str, int] = field(default_factory=dict)
    per_stage: Mapping[str, Mapping[str, int]] = field(default_factory=dict)
    density: float = 0.0

    @property
    def steps(self) -> int:
        """Return the number of simulated timesteps."""
        return int(self.trajectory.steps)

    @property
    def logits(self) -> Any:
        """Return the averaged readout tensor."""
        return self.trajectory.logits

    @property
    def sop(self) -> int:
        """Return the synaptic-operation count."""
        return int(self.counts.get("sop", 0))

    @property
    def mac(self) -> int:
        """Return the dense multiply-accumulate baseline count."""
        return int(self.counts.get("mac", 0))

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-able summary of the run."""
        return {
            "steps": self.steps,
            "density": self.density,
            "counts": dict(self.counts),
            "per_stage": {
                name: dict(values) for name, values in self.per_stage.items()
            },
        }
