"""Abstract base class for exporters of trained SSNTrainer data."""

import os

from snn_interpreter.config import DATA_DIR


class Exporter:
    """Save a trained trainer's data to a file on disk."""

    DEFAULT_FILENAME = ""

    def __init__(self, trainer):
        self._trainer = trainer

    @property
    def trainer(self):
        return self._trainer

    @property
    def output_path(self):
        """Full path under the data dir for this exporter's default file."""
        os.makedirs(DATA_DIR, exist_ok=True)
        return os.path.join(DATA_DIR, self.DEFAULT_FILENAME)

    def export(self, filepath=None):
        """Write output to filepath, defaulting under the data dir."""
        raise NotImplementedError
