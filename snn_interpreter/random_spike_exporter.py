"""Export random spike generation visuals (tutorial 3)."""

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from snn_interpreter.exporter import Exporter


class RandomSpikeVideoExporter(Exporter):
    """Save the random spike train animation as MP4."""

    DEFAULT_FILENAME = "random_spikes.mp4"

    def export(self, filepath=None):
        """Write the random spike sample animation to an MP4."""
        filepath = filepath or self.output_path
        sample = self.trainer.spike_rand
        fig, ax = plt.subplots()
        anim = splt.animator(sample, fig, ax, interval=50)
        anim.save(filepath)
        plt.close(fig)


class RandomSpikeRasterExporter(Exporter):
    """Save the random spike train raster plot."""

    DEFAULT_FILENAME = "random_spikes_raster.png"

    def export(self, filepath=None):
        """Write the raster of the generated random spikes."""
        filepath = filepath or self.output_path
        spike_rand = self.trainer.spike_rand
        num_steps, height, width = spike_rand.size()
        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        splt.raster(
            spike_rand.view(num_steps, -1), ax, s=25, c="black"
        )
        ax.set_title("Input Layer")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)
