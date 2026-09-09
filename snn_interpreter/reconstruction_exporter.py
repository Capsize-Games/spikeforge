"""Export the reconstruction figure comparing spiking frequencies."""

import matplotlib.pyplot as plt

from snn_interpreter.exporter import Exporter
from snn_interpreter.plot_utils import (
    spike_sample,
    time_averaged_image,
)


class ReconstructionExporter(Exporter):
    """Render the gain=1 vs low-gain reconstructions side by side."""

    DEFAULT_FILENAME = "spike_reconstruction.png"

    def export(self, filepath=None):
        """Save a two-panel reconstruction figure to disk."""
        filepath = filepath or self.output_path
        trainer = self.trainer
        self._draw_panel(
            121, time_averaged_image(
                spike_sample(trainer.spike_data), trainer.data_size
            ), "Gain = 1",
        )
        self._draw_panel(
            122, time_averaged_image(
                spike_sample(trainer.spike_data_low_gain), trainer.data_size
            ), f"Gain = {trainer.gain}",
        )
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    @staticmethod
    def _draw_panel(subplot_index, image, title):
        plt.subplot(subplot_index)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
        plt.title(title)
