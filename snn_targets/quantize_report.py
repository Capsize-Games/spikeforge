"""The JSON-able report of applying a target's declared quantization.

``applied`` is true only when weights were actually restricted to a scheme's
levels; a target that declares ``none`` or an unknown scheme leaves the graph
untouched and records the named ``reason`` instead. ``layers`` lists the
per-layer before/after value ranges so the clamping is visible, and the
optional ``drift`` section quantifies how far the quantized trajectory moved
from the original when both were executed by the reference interpreter.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

#: One per-layer before/after range record.
Layer = Mapping[str, Any]
#: The per-layer records attached to a report.
Layers = Tuple[Layer, ...]


@dataclass(frozen=True)
class QuantizationReport:
    """The applied/unapplied outcome of quantizing for one target."""

    target: str
    scheme: str
    applied: bool
    reason: str
    layers: Layers = ()
    drift: Optional[Mapping[str, Any]] = field(default=None)

    def counts(self) -> Dict[str, int]:
        """Return the number of weight-bearing layers quantized."""
        return {"layers": len(self.layers)}

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "target": self.target,
            "scheme": self.scheme,
            "applied": self.applied,
            "reason": self.reason,
            "layers": [dict(item) for item in self.layers],
            "counts": self.counts(),
            "drift": None if self.drift is None else dict(self.drift),
        }
