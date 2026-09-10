"""Run the additional tutorial-1 encoding demos (latency/delta/random).

Produces latency curve + rasters + animation, delta plots, and random
spike visuals under build/.
"""

from snn_interpreter.encoding.delta_trainer import DeltaTrainer
from snn_interpreter.encoding.latency_trainer import LatencyTrainer
from snn_interpreter.encoding.random_spikegen import RandomSpikeGenerator
from snn_interpreter.exporters.delta_exporter import DeltaPlotExporter
from snn_interpreter.exporters.latency_curve_exporter import (
    LatencyCurveExporter,
)
from snn_interpreter.exporters.latency_raster_exporter import (
    LatencyRasterExporter,
)
from snn_interpreter.exporters.latency_video_exporter import (
    LatencyVideoExporter,
)
from snn_interpreter.exporters.random_spike_raster_exporter import (
    RandomSpikeRasterExporter,
)
from snn_interpreter.exporters.random_spike_video_exporter import (
    RandomSpikeVideoExporter,
)


def main() -> None:
    """Train the latency/delta/random encoders and export their visuals."""
    latency_trainer = LatencyTrainer(animation_interval=100)
    LatencyCurveExporter(latency_trainer).export()
    LatencyRasterExporter(latency_trainer).export()
    LatencyVideoExporter(latency_trainer).export(key="clip")

    DeltaPlotExporter(DeltaTrainer()).export()

    random_spikegen = RandomSpikeGenerator(num_steps=100)
    RandomSpikeRasterExporter(random_spikegen).export()
    RandomSpikeVideoExporter(random_spikegen).export()


if __name__ == "__main__":
    main()
