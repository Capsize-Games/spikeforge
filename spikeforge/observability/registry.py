"""Thread-safe in-process counters, gauges, and timers.

The registry is deliberately tiny: it stores scalar counters, last-write
gauges, and per-name timer samples, and :meth:`MetricsRegistry.snapshot`
reduces them to JSON-able data. A lock keeps it safe to update from the
training worker thread while the event loop reads a snapshot.
"""

import threading
from typing import Any, Dict, List

from spikeforge.observability.timer import Timer


def _summarise(samples: List[float]) -> Dict[str, Any]:
    """Return count/total/mean/min/max for one timer's samples."""
    count = len(samples)
    total = float(sum(samples))
    return {
        "count": count,
        "total_seconds": total,
        "mean_seconds": total / count if count else 0.0,
        "min_seconds": min(samples) if samples else 0.0,
        "max_seconds": max(samples) if samples else 0.0,
    }


class MetricsRegistry:
    """Collect counters, gauges, and timer samples for one process."""

    def __init__(self) -> None:
        """Create an empty registry."""
        self._lock = threading.Lock()
        self._counters: Dict[str, float] = {}
        self._gauges: Dict[str, float] = {}
        self._timers: Dict[str, List[float]] = {}

    def counter(self, name: str, amount: float = 1.0) -> None:
        """Add ``amount`` (default one) to the counter ``name``."""
        with self._lock:
            current = self._counters.get(name, 0.0)
            self._counters[name] = current + float(amount)

    def gauge(self, name: str, value: float) -> None:
        """Set the gauge ``name`` to ``value``."""
        with self._lock:
            self._gauges[name] = float(value)

    def observe(self, name: str, seconds: float) -> None:
        """Append one ``seconds`` sample to the timer ``name``."""
        with self._lock:
            self._timers.setdefault(name, []).append(float(seconds))

    def timer(self, name: str) -> Timer:
        """Return a context manager that records into timer ``name``."""
        return Timer(self, name)

    def timers(self) -> Dict[str, List[float]]:
        """Return a copy of the raw timer samples, keyed by name.

        The JSON snapshot only carries per-timer summaries, but a Prometheus
        histogram needs the individual samples to place them in buckets, so
        the exporter reads them through this accessor instead of the private
        attribute.
        """
        with self._lock:
            return {
                name: list(samples) for name, samples in self._timers.items()
            }

    def snapshot(self) -> Dict[str, Any]:
        """Return JSON-able counters, gauges, and summarised timers."""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "timers": {
                    name: _summarise(samples)
                    for name, samples in self._timers.items()
                },
            }

    def reset(self) -> None:
        """Drop every recorded counter, gauge, and timer."""
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._timers.clear()
