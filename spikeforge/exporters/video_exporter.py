"""Export the rate-coded spike train as an MP4 video."""

from typing import Optional

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.plot_utils import spike_sample


class VideoExporter(Exporter):
    """Save the gain=1 spike animation as an MP4 via celluloid."""

    DEFAULT_FILENAME = "spike_mnist_test.mp4"

    def export(self, filepath: Optional[str] = None) -> None:
        """Write the animated spike sample to an MP4 file."""
        filepath = filepath or self.output_path
        fig, ax = plt.subplots()
        sample = spike_sample(self.trainer.spike_data)
        anim = splt.animator(sample, fig, ax, interval=self.trainer.interval)
        anim.save(filepath)
        plt.close(fig)
