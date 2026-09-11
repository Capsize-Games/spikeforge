"""The JSON-able energy/latency report produced by the accounting layer.

``estimate`` is ``True`` for every declared-cost or unavailable report and only
becomes ``False`` when a real device reported its own timing (a ``measured``
block). ``basis`` names where the numbers come from, and an unavailable target
carries ``None`` energy/latency rather than a fabricated value.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple


@dataclass(frozen=True)
class EnergyReport:
    """Accounted operations plus an estimated or measured energy/latency."""

    target: str
    estimate: bool
    basis: str
    timesteps: Optional[int]
    ops: Mapping[str, int]
    efficiency: Mapping[str, Optional[float]]
    energy: Optional[Mapping[str, float]]
    latency: Optional[Mapping[str, float]]
    notes: Tuple[str, ...]
    measured: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-able dict."""
        return {
            "target": self.target,
            "estimate": self.estimate,
            "basis": self.basis,
            "timesteps": self.timesteps,
            "ops": dict(self.ops),
            "efficiency": dict(self.efficiency),
            "energy": None if self.energy is None else dict(self.energy),
            "latency": None if self.latency is None else dict(self.latency),
            "measured": (
                None if self.measured is None else dict(self.measured)
            ),
            "notes": list(self.notes),
        }
