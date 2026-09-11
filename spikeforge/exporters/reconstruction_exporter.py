"""Export the reconstruction figure comparing spiking frequencies."""

from typing import Optional

import matplotlib.pyplot as plt
import torch

from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.plot_utils import (
    spike_sample,
    time_averaged_image,
)


class ReconstructionExporter(Exporter):
    """Render the gain=1 vs low-gain reconstructions side by side."""

    DEFAULT_FILENAME = "spike_reconstruction.png"

    def export(self, filepath: Optional[str] = None) -> None:
        """Save a two-panel reconstruction figure to disk."""
        filepath = filepath or self.output_path
        self._draw_pair()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def _draw_pair(self) -> None:
        """Draw the gain=1 and low-gain reconstruction panels."""
        trainer = self.trainer
        panels = [
            (121, spike_sample(trainer.spike_data), "Gain = 1"),
            (122, spike_sample(trainer.spike_data_low_gain),
             f"Gain = {trainer.gain}"),
        ]
        for index, sample, title in panels:
            self._draw_panel(
                index, time_averaged_image(sample, trainer.data_size),
                title,
            )

    @staticmethod
    def _draw_panel(
        subplot_index: int, image: torch.Tensor, title: str
    ) -> None:
        """Draw one reconstruction image panel."""
        plt.subplot(subplot_index)
        plt.imshow(image, cmap="binary")
        plt.axis("off")
        plt.title(title)
