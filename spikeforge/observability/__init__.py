"""Opt-in structured logging and an in-process metrics registry.

Both surfaces are additive: importing this package has no side effects, the
logging setup only acts when asked (via an environment variable or
``force=True``), and the metrics registry is a plain in-process object whose
:meth:`~spikeforge.observability.registry.MetricsRegistry.snapshot`
returns JSON-able data for the dashboard.
"""

from spikeforge.observability import metrics, persistence
from spikeforge.observability.logging_setup import (
    configure_logging,
    logging_enabled,
    reset_logging,
)
from spikeforge.observability.registry import MetricsRegistry

__all__ = [
    "MetricsRegistry",
    "configure_logging",
    "logging_enabled",
    "metrics",
    "persistence",
    "reset_logging",
]
