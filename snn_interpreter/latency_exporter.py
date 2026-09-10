"""Export latency-coding visualizations (tutorial 2.3)."""

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt
import torch

from snn_interpreter.exporter import Exporter
from snn_interpreter.latency_trainer import convert_to_time
from snn_interpreter.plot_utils import spike_sample


class LatencyCurveExporter(Exporter):
    """Render the input-intensity vs spike-time curve."""

    DEFAULT_FILENAME = "latency_curve.png"

    def export(self, filepath=None):
        """Save the RC latency transfer curve to disk."""
        filepath = filepath or self.output_path
        raw_input = torch.arange(0, 5, 0.05)
        spike_times = convert_to_time(raw_input)
        plt.plot(raw_input, spike_times)
        plt.xlabel("Input Value")
        plt.ylabel("Spike Time (s)")
        plt.savefig(filepath, bbox_inches="tight")
        plt.close()


class LatencyRasterExporter(Exporter):
    """Render rasters of each latency variant as a grid."""

    DEFAULT_FILENAME = "latency_rasters.png"

    def export(self, filepath=None):
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
    def _panel(ax, trainer, key):
        spike_data = trainer.latency_data[key]
        num_steps = spike_data.size(0)
        sample = spike_sample(spike_data)
        splt.raster(sample.reshape(num_steps, -1), ax, s=1.5, c="black")
        ax.set_title(f"Latency ({key})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")


class LatencyVideoExporter(Exporter):
    """Save a latency spike animation as MP4."""

    DEFAULT_FILENAME = "latency_animation.mp4"

    def export(self, filepath=None, key="clip"):
        """Write the animated latency sample to an MP4."""
        filepath = filepath or self.output_path
        sample = spike_sample(self.trainer.latency_data[key])
        fig, ax = plt.subplots()
        anim = splt.animator(sample, fig, ax, interval=self.trainer.interval)
        anim.save(filepath)
        plt.close(fig)
