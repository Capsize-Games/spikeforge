"""Export a latency spike animation as MP4."""

from typing import Optional

import matplotlib.pyplot as plt

from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.plot_utils import spike_sample


class LatencyVideoExporter(Exporter):
    """Save a latency spike animation as MP4."""

    DEFAULT_FILENAME = "latency_animation.mp4"

    def export(
        self, filepath: Optional[str] = None, key: str = "clip"
    ) -> None:
        """Write the animated latency sample to an MP4."""
        import snntorch.spikeplot as splt

        filepath = filepath or self.output_path
        sample = spike_sample(self.trainer.latency_data[key])
        fig, ax = plt.subplots()
        anim = splt.animator(sample, fig, ax, interval=self.trainer.interval)
        anim.save(filepath)
        plt.close(fig)
