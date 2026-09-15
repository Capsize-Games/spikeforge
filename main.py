"""Entry point: train a rate-coded MNIST subset and export all visuals."""

import argparse
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
    """Return the argument parser for the ``spikeforge`` demo CLI."""
    parser = argparse.ArgumentParser(
        prog="spikeforge",
        description=(
            "Train a rate-coded MNIST subset with snnTorch and export "
            "every tutorial visual (video, GIFs, raster, reconstruction) "
            "to build/. Downloads MNIST on first run; this is the "
            "tutorial demo, not a general-purpose training CLI -- see "
            "spikeforge-verify, spikeforge-benchmark, and "
            "documentation/usage.md for the rest of the toolkit."
        ),
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="print the installed spikeforge versions and exit",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    """Parse ``argv``, then build the trainer and write every output."""
    args = _parser().parse_args(argv)

    # ``--version`` resolves the report here rather than through
    # ``spikeforge.version.add_version_flag`` so that neither it nor
    # ``--help`` imports the spikeforge package: this entry point answers
    # ``--help`` in milliseconds on a bare core install, and importing the
    # package would make it pay for torch first.
    if args.version:
        from spikeforge.version import version_report

        print(version_report())
        return

    # Imported lazily so ``--help`` never pays for (or fails on) the heavy
    # snnTorch/matplotlib export stack before argparse has even run.
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

    trainer = SNNTrainerLogger(animation_interval=100)
    VideoExporter(trainer).export()
    SpikeGifExporter(trainer).export()
    ReconstructionExporter(trainer).export()
    RasterExporter(trainer).export()
    PresentationGifExporter(trainer).export()


if __name__ == "__main__":
    main()
