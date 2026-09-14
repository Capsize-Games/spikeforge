"""Export a raster plot of the input layer and one neuron."""

from typing import Optional

import matplotlib.pyplot as plt
import torch

from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.plot_utils import neuron_index, spike_sample


class RasterExporter(Exporter):
    """Draw spike rasters for the input layer and a single neuron."""

    DEFAULT_FILENAME = "spike_raster.png"

    def export(
        self, filepath: Optional[str] = None, neuron_idx: Optional[int] = None
    ) -> None:
        """Save a two-panel raster figure to disk."""
        filepath = filepath or self.output_path

        trainer = self.trainer
        num_steps = trainer.spike_data.size(0)
        sample = spike_sample(trainer.spike_data)
        sample_low = spike_sample(trainer.spike_data_low_gain)
        if neuron_idx is None:
            neuron_idx = neuron_index(sample)

        self._layer_axes(num_steps, sample_low)
        self._neuron_axes(num_steps, sample, neuron_idx)
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()

    def _layer_axes(self, num_steps: int, sample: torch.Tensor) -> None:
        """Draw the full input-layer raster panel."""
        import snntorch.spikeplot as splt

        ax = plt.subplot(2, 1, 1)
        splt.raster(sample.reshape(num_steps, -1), ax, s=1.5, c="black")
        ax.set_title(f"Input Layer (Gain = {self.trainer.gain})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")

    def _neuron_axes(
        self, num_steps: int, sample: torch.Tensor, neuron_idx: int
    ) -> None:
        """Draw the single-neuron raster panel."""
        import snntorch.spikeplot as splt

        ax = plt.subplot(2, 1, 2)
        splt.raster(
            sample.reshape(num_steps, -1)[:, neuron_idx].unsqueeze(1),
            ax,
            s=100,
            c="black",
            marker="|",
        )
        ax.set_title(f"Input Neuron {neuron_idx} (Gain = 1)")
        ax.set_xlabel("Time step")
        ax.set_yticks([])
