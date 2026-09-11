"""A point-in-time capture of the in-process metrics registry.

The snapshot is the durable unit of E1: it pairs a run identifier and a
timestamp with the JSON-able registry snapshot already produced by
:meth:`~snn_interpreter.observability.registry.MetricsRegistry.snapshot`, so a
persisted file can be reloaded without re-deriving anything.
"""

import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class MetricSnapshot:
    """A metrics snapshot plus the run it belongs to and when it was taken."""

    run_id: str
    metrics: Dict[str, Any]
    timestamp: float

    @classmethod
    def capture(
        cls,
        run_id: str,
        metrics: Mapping[str, Any],
        timestamp: Optional[float] = None,
    ) -> "MetricSnapshot":
        """Return a snapshot of ``metrics`` for ``run_id`` at ``timestamp``."""
        taken = time.time() if timestamp is None else float(timestamp)
        return cls(run_id=str(run_id), metrics=dict(metrics), timestamp=taken)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able snapshot payload written to disk."""
        return {
            "run_id": self.run_id,
            "timestamp": self.timestamp,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricSnapshot":
        """Rebuild a snapshot from :meth:`to_dict` output."""
        return cls(
            run_id=str(data.get("run_id", "")),
            metrics=dict(data.get("metrics", {})),
            timestamp=float(data.get("timestamp", 0.0)),
        )
