"""Entry point: train a rate-coded MNIST subset and export all visuals."""

from spikeforge.exporters.presentation_exporter import (
    PresentationGifExporter,
)
from spikeforge.exporters.raster_exporter import RasterExporter
from spikeforge.exporters.reconstruction_exporter import (
    ReconstructionExporter,
)
from spikeforge.exporters.spike_gif_exporter import SpikeGifExporter
from spikeforge.exporters.video_exporter import VideoExporter
from spikeforge.training.logger import SNNTrainerLogger


def main() -> None:
    """Build the trainer and write every output artifact."""
    trainer = SNNTrainerLogger(animation_interval=100)
    VideoExporter(trainer).export()
    SpikeGifExporter(trainer).export()
    ReconstructionExporter(trainer).export()
    RasterExporter(trainer).export()
    PresentationGifExporter(trainer).export()


if __name__ == "__main__":
    main()
