"""Run the additional tutorial-1 encoding demos (latency/delta/random).

Produces latency curve + rasters + animation, delta plots, and random
spike visuals under build/.
"""

from snn_interpreter.delta_exporter import DeltaPlotExporter
from snn_interpreter.delta_trainer import DeltaTrainer
from snn_interpreter.latency_exporter import (
    LatencyCurveExporter,
    LatencyRasterExporter,
    LatencyVideoExporter,
)
from snn_interpreter.latency_trainer import LatencyTrainer
from snn_interpreter.random_spike_exporter import (
    RandomSpikeRasterExporter,
    RandomSpikeVideoExporter,
)
from snn_interpreter.random_spikegen import RandomSpikeGenerator


def main():
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
