"""Export the latency input-intensity vs spike-time curve."""

from typing import Optional

import matplotlib.pyplot as plt
import torch

from spikeforge.encoding.latency_trainer import convert_to_time
from spikeforge.exporters.exporter import Exporter


class LatencyCurveExporter(Exporter):
    """Render the input-intensity vs spike-time curve."""

    DEFAULT_FILENAME = "latency_curve.png"

    def export(self, filepath: Optional[str] = None) -> None:
        """Save the RC latency transfer curve to disk."""
        filepath = filepath or self.output_path
        raw_input = torch.arange(0, 5, 0.05)
        spike_times = convert_to_time(raw_input)
        plt.plot(raw_input, spike_times)
        plt.xlabel("Input Value")
        plt.ylabel("Spike Time (s)")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()
