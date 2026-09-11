"""A target's declared per-operation energy and per-timestep latency.

Values in a bundled table are **declared estimates**, never measurements: the
``measured`` flag is ``false`` for every shipped table and ``source`` names
where the numbers come from. The accounting layer treats a table as an
estimate regardless, and only a device probe that reports its own timing makes
a report ``measured``.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from spikeforge_targets.energy.errors import EnergyCostError

#: Fields every bundled cost table must declare.
REQUIRED_KEYS = ("sop_pj", "mac_pj", "ac_pj", "step_ns", "source", "measured")


@dataclass(frozen=True)
class CostTable:
    """Declared per-op energy (pJ) and per-step latency (ns) for a target."""

    target: str
    sop_pj: float
    mac_pj: float
    ac_pj: float
    step_ns: float
    source: str
    measured: bool = False

    @classmethod
    def from_dict(cls, target: str, data: Mapping[str, Any]) -> "CostTable":
        """Validate ``data`` and build the table for ``target``."""
        missing = [key for key in REQUIRED_KEYS if key not in data]
        if missing:
            joined = ", ".join(missing)
            raise EnergyCostError(target, f"missing keys: {joined}")
        return cls(
            target=target,
            sop_pj=float(data["sop_pj"]),
            mac_pj=float(data["mac_pj"]),
            ac_pj=float(data["ac_pj"]),
            step_ns=float(data["step_ns"]),
            source=str(data["source"]),
            measured=bool(data["measured"]),
        )

    def energy_pj(self, sop: int, mac: int, ac: int) -> Dict[str, float]:
        """Return per-op energy totals in picojoules for one count set.

        ``total_pj`` is the event-driven total (spikes plus neuron updates);
        ``dense_pj`` is the dense baseline alternative. Summing SOP and MAC
        would double count, so the two totals are reported separately.
        """
        sop_energy = sop * self.sop_pj
        mac_energy = mac * self.mac_pj
        ac_energy = ac * self.ac_pj
        return {
            "sop_pj": sop_energy,
            "mac_pj": mac_energy,
            "ac_pj": ac_energy,
            "total_pj": sop_energy + ac_energy,
            "dense_pj": mac_energy + ac_energy,
        }

    def latency_ns(self, steps: int) -> Dict[str, float]:
        """Return per-step and total nominal latency in nanoseconds."""
        return {"step_ns": self.step_ns, "total_ns": steps * self.step_ns}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-able description of the declared table."""
        return {
            "target": self.target,
            "sop_pj": self.sop_pj,
            "mac_pj": self.mac_pj,
            "ac_pj": self.ac_pj,
            "step_ns": self.step_ns,
            "source": self.source,
            "measured": self.measured,
        }
