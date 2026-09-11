"""Read and write persisted metric snapshots as JSON under ``METRICS_DIR``.

One file per run id keeps the layout trivial to inspect: the latest flush for
a run overwrites that run's file through an atomic replace, so a crash during
a write never leaves a half-written snapshot behind.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

from snn_interpreter import config
from snn_interpreter.observability.snapshot import MetricSnapshot

_SAFE = re.compile(r"[^A-Za-z0-9_.-]")


def safe_run_id(run_id: str) -> str:
    """Sanitise a run id into a filesystem-safe stem."""
    cleaned = _SAFE.sub("_", str(run_id).strip())
    return cleaned or "default"


def resolve_root(root: Optional[str] = None) -> str:
    """Resolve the store root: explicit, then env override, then config."""
    if root is not None:
        return root
    return os.environ.get("SNN_METRICS_DIR", config.METRICS_DIR)


class SnapshotStore:
    """Persist and reload :class:`MetricSnapshot` files for one root."""

    def __init__(self, root: Optional[str] = None) -> None:
        """Bind the store to ``root`` (default: ``config.METRICS_DIR``)."""
        self._root = Path(resolve_root(root))

    @property
    def root(self) -> str:
        """Return the directory persisted snapshots live under."""
        return str(self._root)

    def path_for(self, run_id: str) -> Path:
        """Return the snapshot file path for ``run_id``."""
        return self._root / f"{safe_run_id(run_id)}.json"

    def write(self, snapshot: MetricSnapshot) -> str:
        """Write ``snapshot`` atomically and return its path."""
        self._root.mkdir(parents=True, exist_ok=True)
        target = self.path_for(snapshot.run_id)
        temporary = target.with_suffix(".json.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(snapshot.to_dict(), handle, sort_keys=True)
        os.replace(temporary, target)
        return str(target)

    def read(self, run_id: str) -> Optional[MetricSnapshot]:
        """Return the stored snapshot for ``run_id``, or None if absent."""
        path = self.path_for(run_id)
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as handle:
            return MetricSnapshot.from_dict(json.load(handle))

    def list_runs(self) -> List[str]:
        """Return the run ids that currently have a stored snapshot."""
        if not self._root.exists():
            return []
        return sorted(
            entry.stem
            for entry in self._root.iterdir()
            if entry.suffix == ".json"
        )
