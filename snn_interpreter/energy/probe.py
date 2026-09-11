"""Isolated probe for a real device that reports its own energy/latency.

The honesty rule lives here: measured numbers exist only when a real device
backend reports its own timing. None of the bundled simulator targets can do
that, so :func:`measure` returns ``None`` and the account stays a declared
estimate. A backend that grows an energy API is wired in here so callers flip
to ``measured`` without changing their own code.
"""

from typing import Any, Dict, Optional


def measure(target: str) -> Optional[Dict[str, Any]]:
    """Return a device-reported measurement for ``target``, or ``None``.

    ``None`` means no device reported its own timing, so the caller must fall
    back to the declared cost table (an estimate). It never fabricates a
    number to look measured.
    """
    return None


def reason(target: str) -> str:
    """Return why ``target`` has no measured energy on this machine."""
    return (
        f"no device reports measured energy for target {target!r}; "
        "the report is a declared estimate"
    )
