"""Export delta-modulation visualizations (tutorial 2.4)."""

import matplotlib.pyplot as plt
import snntorch.spikeplot as splt

from snn_interpreter.exporter import Exporter


class DeltaPlotExporter(Exporter):
    """Render the fake time series and its delta raster(s)."""

    DEFAULT_FILENAME = "delta_plots.png"

    def export(self, filepath=None):
        """Save the time-series line plot and both raster variants."""
        filepath = filepath or self.output_path
        trainer = self.trainer
        fig = plt.figure(facecolor="w", figsize=(8, 8))

        ax_series = fig.add_subplot(3, 1, 1)
        ax_series.plot(trainer.data)
        ax_series.set_title("Some fake time-series data")
        ax_series.set_xlabel("Time step")
        ax_series.set_ylabel("Voltage (mV)")

        self._raster_panel(fig, 2, trainer.spike_data, "on-spikes")
        self._raster_panel(fig, 3, trainer.spike_data_off, "off-spikes")

        plt.tight_layout()
        plt.savefig(filepath, bbox_inches="tight")
        plt.close(fig)

    @staticmethod
    def _raster_panel(fig, position, spike_data, title):
        ax = fig.add_subplot(3, 1, position)
        splt.raster(spike_data, ax, c="black")
        ax.set_title(f"Delta ({title})")
        ax.set_xlabel("Time step")
        ax.set_yticks([])
        ax.set_xlim(0, len(spike_data))
