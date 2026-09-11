"""Abstract base class for exporters of trained SSNTrainer data."""

import os
from typing import Any, Optional

from spikeforge.config import DATA_DIR


class Exporter:
    """Save a trained trainer's data to a file on disk."""

    DEFAULT_FILENAME: str = ""

    def __init__(self, trainer: Any) -> None:
        """Store the trainer or spike source this exporter renders."""
        self._trainer = trainer

    @property
    def trainer(self) -> Any:
        """Return the wrapped trainer or spike source."""
        return self._trainer

    @property
    def output_path(self) -> str:
        """Full path under the data dir for this exporter's default file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        return os.path.join(DATA_DIR, self.DEFAULT_FILENAME)

    def export(self, filepath: Optional[str] = None) -> None:
        """Write output to filepath, defaulting under the data dir."""
        raise NotImplementedError
