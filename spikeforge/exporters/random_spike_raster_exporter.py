"""Export the random spike train raster plot."""

from typing import Optional

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from spikeforge.exporters.exporter import Exporter


class RandomSpikeRasterExporter(Exporter):
    """Save the random spike train raster plot."""

    DEFAULT_FILENAME = "random_spikes_raster.png"

    def export(self, filepath: Optional[str] = None) -> None:
        """Write the raster of the generated random spikes."""
        filepath = filepath or self.output_path
        spike_rand = self.trainer.spike_rand
        num_steps = spike_rand.size(0)
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(spike_rand.view(num_steps, -1), ax, s=25, c="black")
        ax.set_title("Input Layer")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
