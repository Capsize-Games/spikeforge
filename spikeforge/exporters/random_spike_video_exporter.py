"""Export the random spike train animation as MP4."""

from typing import Optional

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from spikeforge.exporters.exporter import Exporter


class RandomSpikeVideoExporter(Exporter):
    """Save the random spike train animation as MP4."""

    DEFAULT_FILENAME = "random_spikes.mp4"

    def export(self, filepath: Optional[str] = None) -> None:
        """Write the random spike sample animation to an MP4."""
        filepath = filepath or self.output_path
        sample = self.trainer.spike_rand
        fig, ax = plt.subplots()
        anim = splt.animator(sample, fig, ax, interval=50)
        anim.save(filepath)
        plt.close(fig)
