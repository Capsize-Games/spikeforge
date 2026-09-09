"""Entry point: train a rate-coded MNIST subset and export all visuals."""

from snn_interpreter.logger import SNNTrainerLogger
from snn_interpreter.presentation_exporter import PresentationGifExporter
from snn_interpreter.raster_exporter import RasterExporter
from snn_interpreter.reconstruction_exporter import ReconstructionExporter
from snn_interpreter.spike_gif_exporter import SpikeGifExporter
from snn_interpreter.video_exporter import VideoExporter


def main():
    """Build the trainer and write every output artifact."""
    trainer = SNNTrainerLogger(animation_interval=100)
    VideoExporter(trainer).export()
    SpikeGifExporter(trainer).export()
    ReconstructionExporter(trainer).export()
    RasterExporter(trainer).export()
    PresentationGifExporter(trainer).export()


if __name__ == "__main__":
    main()
