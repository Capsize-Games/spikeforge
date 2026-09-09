"""Abstract base class for exporters of trained SSNTrainer data."""


class Exporter:
    """Save a trained trainer's data to a file on disk."""

    DEFAULT_FILENAME = ""

    def __init__(self, trainer):
        self._trainer = trainer

    @property
    def trainer(self):
        return self._trainer

    def export(self, filepath=None):
        """Write output to filepath, defaulting to DEFAULT_FILENAME."""
        raise NotImplementedError
