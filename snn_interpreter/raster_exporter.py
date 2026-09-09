"""Export a raster plot of the input layer and one neuron."""

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from snn_interpreter.exporter import Exporter
from snn_interpreter.plot_utils import (
    neuron_index,
    spike_sample,
)


class RasterExporter(Exporter):
    """Draw spike rasters for the input layer and a single neuron."""

    DEFAULT_FILENAME = "spike_raster.png"

    def export(self, filepath=None, neuron_idx=None):
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

    def _layer_axes(self, num_steps, sample):
        ax = plt.subplot(2, 1, 1)
        splt.raster(
            sample.reshape(num_steps, -1), ax, s=1.5, c="black"
        )
        ax.set_title(f"Input Layer (Gain = {self.trainer.gain})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")

    def _neuron_axes(self, num_steps, sample, neuron_idx):
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
