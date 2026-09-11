"""A context manager that records its duration into a metrics registry."""

import time
from types import TracebackType
from typing import Any, Optional, Type


class Timer:
    """Time a ``with`` block and emit its duration when the block exits."""

    def __init__(self, registry: Any, name: str) -> None:
        """Bind the timer to ``registry`` under the sample name ``name``."""
        self._registry = registry
        self._name = name
        self._start = 0.0

    def __enter__(self) -> "Timer":
        """Start the clock and return this timer."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        """Record the elapsed seconds, even when the block raised."""
        elapsed = time.perf_counter() - self._start
        self._registry.observe(self._name, elapsed)
