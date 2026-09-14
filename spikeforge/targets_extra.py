"""Lazy-import boundary for the optional ``spikeforge-targets`` package.

The core ``spikeforge`` distribution ships a few surfaces that only make
sense once ``spikeforge-targets`` is installed: the deployment subcommands on
``spikeforge-verify`` and opt-in event-driven energy accounting in the
benchmark harness. ``spikeforge-targets`` is not a hard dependency of core
(it ships separately so a headless install stays light), so importing it at
module load would break basic, non-deployment invocations -- including
``--help`` -- on a plain ``pip install spikeforge``. Callers that need it go
through :func:`require` instead, which raises a typed error naming the
install command rather than a bare ``ModuleNotFoundError``.
"""

from importlib import import_module
from typing import Any


class TargetsExtraMissingError(RuntimeError):
    """Raised when a ``spikeforge-targets``-backed feature runs without it."""

    def __init__(self) -> None:
        """Build a message naming the install command."""
        super().__init__(
            "spikeforge-targets is required for this feature; install it "
            'with: pip install spikeforge-targets (or: pip install '
            '"spikeforge[all]")'
        )


def available() -> bool:
    """Return True when ``spikeforge_targets`` can be imported."""
    try:
        import_module("spikeforge_targets")
    except ImportError:
        return False
    return True


def require(dotted_module: str) -> Any:
    """Import and return ``dotted_module``, or raise the typed error.

    ``dotted_module`` is expected to start with ``spikeforge_targets``.
    """
    try:
        return import_module(dotted_module)
    except ImportError as exc:
        raise TargetsExtraMissingError() from exc
