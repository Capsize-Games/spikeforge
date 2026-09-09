"""Abstract base class for exporters of trained SSNTrainer data."""

import os

BUILD_DIR = "build"


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
        """Full path under build/ for this exporter's default file."""
        os.makedirs(BUILD_DIR, exist_ok=True)
        return os.path.join(BUILD_DIR, self.DEFAULT_FILENAME)

    def export(self, filepath=None):
        """Write output to filepath, defaulting under the build dir."""
        raise NotImplementedError
