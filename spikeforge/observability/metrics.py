"""The process-wide metrics registry and its convenience functions.

``spikeforge.observability.metrics`` exposes one shared
:class:`~spikeforge.observability.registry.MetricsRegistry` so callers
can instrument a hot path with a single function call and the server can read
a snapshot without threading a registry through every layer.
"""

from typing import Any, Dict

from spikeforge.observability.registry import MetricsRegistry
from spikeforge.observability.timer import Timer

_registry = MetricsRegistry()


def registry() -> MetricsRegistry:
    """Return the process-wide metrics registry."""
    return _registry


def counter(name: str, amount: float = 1.0) -> None:
    """Add ``amount`` to counter ``name``."""
    _registry.counter(name, amount)


def gauge(name: str, value: float) -> None:
    """Set gauge ``name`` to ``value``."""
    _registry.gauge(name, value)


def observe(name: str, seconds: float) -> None:
    """Record one timer sample ``seconds`` under ``name``."""
    _registry.observe(name, seconds)


def timer(name: str) -> Timer:
    """Return a context manager recording its duration under ``name``."""
    return _registry.timer(name)


def snapshot() -> Dict[str, Any]:
    """Return a JSON-able snapshot of every recorded metric."""
    return _registry.snapshot()


def reset() -> None:
    """Clear every recorded metric on the shared registry."""
    _registry.reset()
