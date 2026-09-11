"""The quantized graph produced by applying a target's quantization.

``graph`` is a fresh NIR graph whose weight-bearing nodes are restricted to
the target's declared scheme's levels, and ``report`` names whether that was
applied, what each layer's range became, and (when a spike fixture was given)
the measured drift. The original graph is never mutated, so a caller can hold
both the before and after renderings.
"""

from dataclasses import dataclass
from typing import Any

from spikeforge_targets.quantize_report import QuantizationReport


@dataclass(frozen=True)
class QuantizationResult:
    """A quantized graph paired with the report describing the change."""

    graph: Any
    report: QuantizationReport

    def applied(self) -> bool:
        """Return True when the graph's weights were actually quantized."""
        return self.report.applied
