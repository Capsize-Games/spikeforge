"""Hook the live metrics registry to a durable snapshot store.

Persistence is opt-in: with ``SPIKEFORGE_METRICS_PERSIST`` unset the
default instance is disabled and :func:`flush` is a no-op that writes
nothing, so the existing
in-memory behaviour is byte-for-byte unchanged. Reads (:func:`load`) never
mutate the live registry, and any failure to write is reported rather than
raised, because metrics must never take down the process they measure.
"""

import os
from typing import Any, Dict, Optional

from spikeforge.observability import metrics
from spikeforge.observability.snapshot import MetricSnapshot
from spikeforge.observability.store import SnapshotStore

#: Environment variable that opts a process into snapshot persistence.
ENV_FLAG = "SPIKEFORGE_METRICS_PERSIST"

#: Run id used when a caller does not supply one.
DEFAULT_RUN_ID = "default"

_TRUTHY = {"1", "true", "yes", "on"}


def _env_enabled() -> bool:
    """Return True when ``SPIKEFORGE_METRICS_PERSIST`` is truthy."""
    return os.environ.get(ENV_FLAG, "").strip().lower() in _TRUTHY


class MetricsPersistence:
    """Snapshot the shared registry to disk and reload previous snapshots."""

    def __init__(
        self,
        root: Optional[str] = None,
        enabled: Optional[bool] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Bind a store, resolving ``enabled`` from the environment if None."""
        self._store = SnapshotStore(root)
        self._enabled = _env_enabled() if enabled is None else bool(enabled)
        self._run_id = run_id or DEFAULT_RUN_ID
        self._last_flush: Optional[float] = None

    @property
    def enabled(self) -> bool:
        """Return whether flushing is currently opted into."""
        return self._enabled

    @property
    def root(self) -> str:
        """Return the directory snapshots are written under."""
        return self._store.root

    @property
    def run_id(self) -> str:
        """Return the run id this instance flushes under by default."""
        return self._run_id

    def flush(self, run_id: Optional[str] = None) -> Optional[MetricSnapshot]:
        """Persist the current registry snapshot, or no-op when disabled."""
        if not self._enabled:
            return None
        snapshot = MetricSnapshot.capture(
            run_id or self._run_id, metrics.snapshot()
        )
        try:
            self._store.write(snapshot)
        except OSError:
            return None
        self._last_flush = snapshot.timestamp
        return snapshot

    def load(self, run_id: Optional[str] = None) -> Optional[MetricSnapshot]:
        """Return a stored snapshot without touching the live registry."""
        return self._store.read(run_id or self._run_id)

    def status(self) -> Dict[str, Any]:
        """Return whether persistence is on, where it writes, and when last."""
        return {
            "enabled": self._enabled,
            "root": self._store.root,
            "run_id": self._run_id,
            "last_flush": self._last_flush,
        }


_default: Optional[MetricsPersistence] = None


def default() -> MetricsPersistence:
    """Return the process-wide persistence instance, creating it lazily."""
    global _default
    if _default is None:
        _default = MetricsPersistence()
    return _default


def reset_default() -> None:
    """Drop the cached instance so a changed environment is re-read."""
    global _default
    _default = None


def flush(run_id: Optional[str] = None) -> Optional[MetricSnapshot]:
    """Persist the shared registry's snapshot via the default instance."""
    return default().flush(run_id)


def load(run_id: Optional[str] = None) -> Optional[MetricSnapshot]:
    """Load a persisted snapshot via the default instance."""
    return default().load(run_id)


def status() -> Dict[str, Any]:
    """Return the default instance's persistence status."""
    return default().status()
