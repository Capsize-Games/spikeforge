"""Run the additional tutorial-1 encoding demos (latency/delta/random).

Produces latency curve + rasters + animation, delta plots, and random
spike visuals under build/.
"""

import argparse
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``spikeforge-encodings`` CLI."""
    return argparse.ArgumentParser(
        prog="spikeforge-encodings",
        description=(
            "Train the latency/delta/random tutorial encoders and export "
            "their visuals to build/. Downloads MNIST on first run; this "
            "is the tutorial demo, not a general-purpose encoding CLI."
        ),
    )


def main(argv: Optional[List[str]] = None) -> None:
    """Parse ``argv``, then train the encoders and export their visuals."""
    _parser().parse_args(argv)

    # Imported lazily so ``--help`` never pays for (or fails on) the heavy
    # snnTorch/matplotlib export stack before argparse has even run.
    from spikeforge.encoding.delta_trainer import DeltaTrainer
    from spikeforge.encoding.latency_trainer import LatencyTrainer
    from spikeforge.encoding.random_spikegen import RandomSpikeGenerator
    from spikeforge.exporters.delta_exporter import DeltaPlotExporter
    from spikeforge.exporters.latency_curve_exporter import (
        LatencyCurveExporter,
    )
    from spikeforge.exporters.latency_raster_exporter import (
        LatencyRasterExporter,
    )
    from spikeforge.exporters.latency_video_exporter import (
        LatencyVideoExporter,
    )
    from spikeforge.exporters.random_spike_raster_exporter import (
        RandomSpikeRasterExporter,
    )
    from spikeforge.exporters.random_spike_video_exporter import (
        RandomSpikeVideoExporter,
    )

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
