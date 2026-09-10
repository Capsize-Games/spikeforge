"""Entry point: train a rate-coded MNIST subset and export all visuals."""

from snn_interpreter.exporters.presentation_exporter import (
    PresentationGifExporter,
)
from snn_interpreter.exporters.raster_exporter import RasterExporter
from snn_interpreter.exporters.reconstruction_exporter import (
    ReconstructionExporter,
)
from snn_interpreter.exporters.spike_gif_exporter import SpikeGifExporter
from snn_interpreter.exporters.video_exporter import VideoExporter
from snn_interpreter.training.logger import SNNTrainerLogger


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
