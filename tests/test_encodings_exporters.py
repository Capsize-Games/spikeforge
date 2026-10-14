"""Smoke tests for every tutorial exporter (issue #117).

Each exporter in :mod:`spikeforge.exporters` is instantiated at tiny
settings and run into ``tmp_path``; the output must exist and be
non-empty. Raster/video exporters draw through ``snntorch.spikeplot``,
which needs the undeclared ``pandas`` runtime (CI installs it; bare
checkouts skip with the reason below). MP4 exporters additionally need
an ``ffmpeg`` binary. The Agg backend is selected before any pyplot
import so the suite stays headless.
"""

import importlib
import inspect
import pkgutil
import shutil
from pathlib import Path
from typing import Any, Iterator, Set

import matplotlib
import pytest

import spikeforge.exporters
from spikeforge.config import DATA_DIR
from spikeforge.encoding.delta_trainer import DeltaTrainer
from spikeforge.encoding.latency_trainer import LatencyTrainer
from spikeforge.encoding.random_spikegen import RandomSpikeGenerator
from spikeforge.exporters.delta_exporter import DeltaPlotExporter
from spikeforge.exporters.exporter import Exporter
from spikeforge.exporters.latency_curve_exporter import (
    LatencyCurveExporter,
)
from spikeforge.exporters.latency_raster_exporter import (
    LatencyRasterExporter,
)
from spikeforge.exporters.latency_video_exporter import (
    LatencyVideoExporter,
)
from spikeforge.exporters.presentation_exporter import (
    PresentationGifExporter,
)
from spikeforge.exporters.random_spike_raster_exporter import (
    RandomSpikeRasterExporter,
)
from spikeforge.exporters.random_spike_video_exporter import (
    RandomSpikeVideoExporter,
)
from spikeforge.exporters.raster_exporter import RasterExporter
from spikeforge.exporters.reconstruction_exporter import (
    ReconstructionExporter,
)
from spikeforge.exporters.spike_gif_exporter import SpikeGifExporter
from spikeforge.exporters.video_exporter import VideoExporter
from spikeforge.training.trainer import SNNTrainer

#: Reason when the spikeplot stack is unavailable in the environment.
_NO_SPIKEPLOT = (
    "snntorch.spikeplot needs pandas, which core does not declare "
    "(issue #117 follow-up: declare it or drop spikeplot)"
)


@pytest.fixture(autouse=True)
def _headless_backend() -> Iterator[None]:
    """Select Agg so figure rendering never needs a display."""
    matplotlib.use("Agg")
    yield



def _tiny_trainer() -> SNNTrainer:
    """Return a rate-coding trainer on a four-sample MNIST slice."""
    return SNNTrainer(
        data_path=DATA_DIR, subset=4, batch_size=2,
        vectorization_num_steps=2, do_shuffle=False,
    )


def _tiny_latency() -> LatencyTrainer:
    """Return a latency trainer on a four-sample MNIST slice."""
    # Steps stay at the tutorial default: the RC spike times of a real
    # batch reach ~58, and spikegen.latency rejects smaller horizons.
    return LatencyTrainer(
        latency_steps=100, data_path=DATA_DIR, subset=4, batch_size=2,
        vectorization_num_steps=2, do_shuffle=False,
    )


def _needs_spikeplot() -> None:
    """Skip the test unless ``snntorch.spikeplot`` imports."""
    pytest.importorskip("snntorch.spikeplot", reason=_NO_SPIKEPLOT)


def _needs_ffmpeg() -> None:
    """Skip the test unless an ``ffmpeg`` binary is on PATH."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("MP4 export needs an ffmpeg binary")


def _assert_exported(exporter: Exporter, path: Path, *args: Any) -> None:
    """Run ``exporter`` into ``path`` and require a non-empty file."""
    exporter.export(str(path), *args)
    assert path.exists()
    assert path.stat().st_size > 0


def _discovered_exporters() -> Set[str]:
    """Return every ``Exporter`` subclass defined in the package."""
    names: Set[str] = set()
    package_path = Path(spikeforge.exporters.__file__).parent
    for module in pkgutil.iter_modules([str(package_path)]):
        imported = importlib.import_module(
            f"spikeforge.exporters.{module.name}"
        )
        for name, member in inspect.getmembers(imported, inspect.isclass):
            if (
                issubclass(member, Exporter)
                and member is not Exporter
                and member.__module__ == imported.__name__
            ):
                names.add(name)
    return names


def test_every_exporter_has_a_case_here() -> None:
    """Adding an exporter without a smoke case fails the suite."""
    cases = {
        "DeltaPlotExporter",
        "LatencyCurveExporter",
        "LatencyRasterExporter",
        "LatencyVideoExporter",
        "PresentationGifExporter",
        "RandomSpikeRasterExporter",
        "RandomSpikeVideoExporter",
        "RasterExporter",
        "ReconstructionExporter",
        "SpikeGifExporter",
        "VideoExporter",
    }
    assert _discovered_exporters() == cases


def test_delta_plots_render_offline(tmp_path: Path) -> None:
    """Delta line plot plus both raster variants render to PNG."""
    _needs_spikeplot()
    exporter = DeltaPlotExporter(DeltaTrainer())
    _assert_exported(exporter, tmp_path / "delta.png")


def test_latency_curve_renders_offline(tmp_path: Path) -> None:
    """The RC transfer curve renders without any trainer state."""
    exporter = LatencyCurveExporter(DeltaTrainer())
    _assert_exported(exporter, tmp_path / "curve.png")


def test_random_spike_raster_renders_offline(tmp_path: Path) -> None:
    """A tiny random train renders its raster to PNG."""
    _needs_spikeplot()
    source = RandomSpikeGenerator(num_steps=4, size=(8, 8))
    exporter = RandomSpikeRasterExporter(source)
    _assert_exported(exporter, tmp_path / "random.png")


def test_random_spike_video_renders_offline(tmp_path: Path) -> None:
    """A tiny random train renders its animation to MP4."""
    _needs_spikeplot()
    _needs_ffmpeg()
    source = RandomSpikeGenerator(num_steps=4, size=(8, 8))
    exporter = RandomSpikeVideoExporter(source)
    _assert_exported(exporter, tmp_path / "random.mp4")


@pytest.mark.network
def test_latency_rasters_render(tmp_path: Path) -> None:
    """All four latency variants render their raster grid to PNG."""
    _needs_spikeplot()
    exporter = LatencyRasterExporter(_tiny_latency())
    _assert_exported(exporter, tmp_path / "latency.png")


@pytest.mark.network
def test_latency_video_renders(tmp_path: Path) -> None:
    """The clip-variant latency animation renders to MP4."""
    _needs_spikeplot()
    _needs_ffmpeg()
    exporter = LatencyVideoExporter(_tiny_latency())
    _assert_exported(exporter, tmp_path / "latency.mp4", "clip")


@pytest.mark.network
def test_rate_raster_renders(tmp_path: Path) -> None:
    """The input-layer and single-neuron rasters render to PNG."""
    _needs_spikeplot()
    exporter = RasterExporter(_tiny_trainer())
    _assert_exported(exporter, tmp_path / "raster.png")


@pytest.mark.network
def test_reconstruction_renders(tmp_path: Path) -> None:
    """The gain-comparison reconstruction renders to PNG."""
    exporter = ReconstructionExporter(_tiny_trainer())
    _assert_exported(exporter, tmp_path / "reconstruction.png")


@pytest.mark.network
def test_spike_gif_renders(tmp_path: Path) -> None:
    """The rate-coded train renders its looping GIF."""
    exporter = SpikeGifExporter(_tiny_trainer())
    _assert_exported(exporter, tmp_path / "spikes.gif")


@pytest.mark.network
def test_presentation_gif_renders(tmp_path: Path) -> None:
    """The full visual tour renders its presentation GIF."""
    exporter = PresentationGifExporter(_tiny_trainer())
    _assert_exported(exporter, tmp_path / "tour.gif", 10)


@pytest.mark.network
def test_rate_video_renders(tmp_path: Path) -> None:
    """The gain=1 spike animation renders to MP4."""
    _needs_spikeplot()
    _needs_ffmpeg()
    exporter = VideoExporter(_tiny_trainer())
    _assert_exported(exporter, tmp_path / "spikes.mp4")
