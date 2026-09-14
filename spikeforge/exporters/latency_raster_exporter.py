"""Export latency-coding raster panels for each variant."""

from typing import Any, Optional

import matplotlib.pyplot as plt
import torch

from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.plot_utils import spike_sample


class LatencyRasterExporter(Exporter):
    """Render rasters of each latency variant as a grid."""

    DEFAULT_FILENAME = "latency_rasters.png"

    def export(self, filepath: Optional[str] = None) -> None:
        """Save four latency raster panels (base/linear/norm/clip)."""
        filepath = filepath or self.output_path
        trainer = self.trainer
        keys = ["base", "linear", "normalized", "clip"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, key in zip(axes.ravel(), keys):
            self._panel(ax, trainer, key)
        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _panel(ax: Any, trainer: Any, key: str) -> None:
        """Draw one latency raster panel."""
        import snntorch.spikeplot as splt

        spike_data: torch.Tensor = trainer.latency_data[key]
        num_steps = spike_data.size(0)
        sample = spike_sample(spike_data)
        splt.raster(sample.reshape(num_steps, -1), ax, s=1.5, c="black")
        ax.set_title(f"Latency ({key})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")
