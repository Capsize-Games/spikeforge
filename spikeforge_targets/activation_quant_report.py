"""The JSON-able report of simulated activation/membrane quantization.

``applied`` is true only when a supported scheme was actually taken; an
unknown scheme is named through ``reason`` and left unapplied rather than
rounded to the nearest supported grid. ``layers`` lists one record per stage
(ranges before and after, and the max/mean absolute error the fixed-point
grid introduced), and ``steps`` counts how many timesteps the hook observed.
Its shape mirrors
:class:`spikeforge_targets.quantize_report.QuantizationReport`, so a caller
can render weight and activation quantization together.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

#: One per-stage error/range record.
Layer = Mapping[str, Any]
#: The per-stage records attached to a report.
Layers = Tuple[Layer, ...]


@dataclass(frozen=True)
class ActivationQuantizationReport:
    """The applied/unapplied outcome of simulated activation quantization."""

    scheme: str
    bits: int
    target: str
    applied: bool
    reason: str
    layers: Layers = ()
    steps: int = 0
    calibration: Optional[Mapping[str, Any]] = field(default=None)

    def counts(self) -> Dict[str, int]:
        """Return the number of quantized stages and observed steps."""
        return {"layers": len(self.layers), "steps": self.steps}

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "scheme": self.scheme,
            "bits": self.bits,
            "target": self.target,
            "applied": self.applied,
            "reason": self.reason,
            "layers": [dict(item) for item in self.layers],
            "counts": self.counts(),
            "calibration": (
                None
                if self.calibration is None
                else dict(self.calibration)
            ),
        }
