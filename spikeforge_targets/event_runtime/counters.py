"""Operation tallies for the event-driven runtime.

The three counts follow the plan's semantics: an **AC** is one accumulate
(one per output element per step), a **SOP** is one synaptic operation (one
per active input-output pair), and a **MAC** is the dense baseline (one per
input-output pair per step). ``timesteps`` is the number of temporal steps.
"""

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SynapticCounter:
    """Accumulate synaptic, dense-multiply, accumulate, and timestep counts."""

    sop: int = 0
    mac: int = 0
    ac: int = 0
    timesteps: int = 0
    per_stage: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, stage: str, sop: int, mac: int, ac: int) -> None:
        """Add one stage's operations for a single step."""
        self.sop += int(sop)
        self.mac += int(mac)
        self.ac += int(ac)
        entry = self.per_stage.setdefault(
            stage, {"sop": 0, "mac": 0, "ac": 0}
        )
        entry["sop"] += int(sop)
        entry["mac"] += int(mac)
        entry["ac"] += int(ac)

    def tick(self) -> None:
        """Count one elapsed timestep."""
        self.timesteps += 1

    def efficiency(self) -> float:
        """Return ``sop / mac`` (0 without a dense MAC baseline)."""
        return 0.0 if self.mac == 0 else self.sop / self.mac

    def counts(self) -> Dict[str, int]:
        """Return the flat scalar tallies."""
        return {
            "sop": self.sop,
            "mac": self.mac,
            "ac": self.ac,
            "timesteps": self.timesteps,
        }

    def stages(self) -> Dict[str, Dict[str, int]]:
        """Return a detached copy of the per-stage tallies."""
        return {name: dict(values) for name, values in self.per_stage.items()}
