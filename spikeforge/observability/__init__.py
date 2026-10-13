"""Opt-in structured logging and an in-process metrics registry.

Both surfaces are additive: importing this package has no side effects, the
logging setup only acts when asked (via an environment variable or
``force=True``), and the metrics registry is a plain in-process object whose
:meth:`~spikeforge.observability.registry.MetricsRegistry.snapshot`
returns JSON-able data for the dashboard.

``metrics`` / ``persistence`` / ``prometheus`` are resolved on first access
(PEP 562) instead of at import time, so the logging surface — and the
:class:`~spikeforge.observability.logging_setup.configure_logging` entry point
that uses the shared ``capsize_commons`` formatter — is importable without
dragging in the metrics stack.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

from spikeforge.observability.logging_setup import (
    configure_logging,
    logging_enabled,
    reset_logging,
)
from spikeforge.observability.registry import MetricsRegistry

if TYPE_CHECKING:
    from spikeforge.observability import metrics, persistence, prometheus

__all__ = [
    "MetricsRegistry",
    "configure_logging",
    "logging_enabled",
    "metrics",
    "persistence",
    "prometheus",
    "reset_logging",
]

#: Submodule name -> import path, resolved on first attribute access.
_LAZY_SUBMODULES = {
    "metrics": "spikeforge.observability.metrics",
    "persistence": "spikeforge.observability.persistence",
    "prometheus": "spikeforge.observability.prometheus",
}


def __getattr__(name: str) -> Any:
    """Import a lazier observability submodule on first access (PEP 562)."""
    try:
        module_name = _LAZY_SUBMODULES[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    module = import_module(module_name)
    globals()[name] = module
    return module
