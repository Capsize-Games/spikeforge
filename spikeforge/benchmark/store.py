"""File-based JSON store for recorded benchmark runs.

A store is a directory of ``<run_id>.json`` files, one per run. It follows the
same convention as ``MODEL_DIR``/``DATA_DIR``: the directory defaults to
``SPIKEFORGE_BENCHMARK_DIR`` (otherwise ``<DATA_DIR>/benchmarks``) and can be
overridden per instance for tests or side-by-side suites.
"""

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from spikeforge.config import DATA_DIR

#: Characters kept verbatim when slugging a label into a run id.
_ID_SAFE = "-_"
#: Longest label slug embedded in a run id.
_MAX_LABEL = 40


def default_directory() -> str:
    """Return the configured benchmark-results directory."""
    return os.environ.get(
        "SPIKEFORGE_BENCHMARK_DIR", os.path.join(DATA_DIR, "benchmarks")
    )


def _slug(label: Optional[str]) -> str:
    """Return a filename-safe slug for ``label`` (``run`` when empty)."""
    if not label:
        return "run"
    safe = "".join(
        char if char.isalnum() or char in _ID_SAFE else "-" for char in label
    ).strip("-_")
    return safe[:_MAX_LABEL] or "run"


class BenchmarkStore:
    """Read and write benchmark run records under one directory."""

    def __init__(self, directory: Optional[str] = None) -> None:
        """Bind the store to ``directory``, or to the configured default."""
        self._directory = Path(directory or default_directory())

    @property
    def directory(self) -> Path:
        """Return the directory this store reads and writes."""
        return self._directory

    def path_for(self, run_id: str) -> Path:
        """Return the JSON path for ``run_id``."""
        return self._directory / f"{run_id}.json"

    def save(
        self,
        record: Mapping[str, Any],
        label: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Write ``record`` (adding its id/label) and return the run id."""
        self._directory.mkdir(parents=True, exist_ok=True)
        identifier = run_id or self._new_id(label)
        payload = dict(record)
        payload["run_id"] = identifier
        payload["label"] = label
        self.path_for(identifier).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return identifier

    def load(self, run_id: str) -> Dict[str, Any]:
        """Return the stored record for ``run_id``.

        Raises ``FileNotFoundError`` so callers can map a missing run to a
        non-zero CLI exit instead of a traceback.
        """
        path = self.path_for(run_id)
        if not path.exists():
            raise FileNotFoundError(f"no benchmark run {run_id!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return a summary of every stored run, newest first."""
        summaries = [self._summary(path) for path in self._paths()]
        usable = [item for item in summaries if item is not None]
        return sorted(
            usable, key=lambda item: item["created_at"], reverse=True
        )

    def _paths(self) -> List[Path]:
        """Return every run JSON path under the directory."""
        if not self._directory.exists():
            return []
        return sorted(self._directory.glob("*.json"))

    def _summary(self, path: Path) -> Optional[Dict[str, Any]]:
        """Return a small summary for one file, or None when unreadable."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return {
            "run_id": data.get("run_id", path.stem),
            "label": data.get("label"),
            "created_at": float(data.get("created_at", 0.0)),
            "config": data.get("config", {}),
            "result_count": len(data.get("results", [])),
        }

    def _new_id(self, label: Optional[str]) -> str:
        """Return a fresh, chronologically sortable run id."""
        return f"{int(time.time() * 1000)}-{_slug(label)}"
