"""Export a combined looping presentation GIF of all visuals."""

from typing import Any, List, Optional, Tuple

import snntorch.spikeplot as splt
from PIL import Image

from snn_interpreter.exporters.exporter import Exporter
from snn_interpreter.exporters.plot_utils import (
    fig_to_image,
    new_fig,
    save_gif,
    spike_sample,
    time_averaged_image,
)


class PresentationGifExporter(Exporter):
    """Assemble title, raster, reconstruction and animation slides."""

    DEFAULT_FILENAME = "spike_presentation.gif"
    SLIDE_MS = 1800

    def export(
        self,
        filepath: Optional[str] = None,
        slide_duration_ms: Optional[int] = None,
    ) -> None:
        """Save the full visual tour as an infinitely looping GIF."""
        filepath = filepath or self.output_path
        if slide_duration_ms is None:
            slide_duration_ms = self.SLIDE_MS
        frames, durations = self._assemble(slide_duration_ms=slide_duration_ms)
        save_gif(filepath, frames, durations)

    def _assemble(
        self, slide_duration_ms: int
    ) -> Tuple[List[Image.Image], List[int]]:
        """Return (frames, durations) for every slide and animation."""
        trainer = self.trainer
        sample = spike_sample(trainer.spike_data)
        sample_low = spike_sample(trainer.spike_data_low_gain)

        frames: List[Image.Image] = []
        durations: List[int] = []
        for image in self._slides(sample, sample_low, trainer):
            frames.append(image)
            durations.append(slide_duration_ms)
        anim_frames, anim_durations = self._animation_frames(
            sample, trainer.interval
        )
        frames.extend(anim_frames)
        durations.extend(anim_durations)
        return frames, durations

    def _slides(
        self, sample: Any, sample_low: Any, trainer: Any
    ) -> List[Image.Image]:
        """Return the ordered static slide images."""
        return [
            self._title_slide(),
            self._raster_slide(sample_low, trainer),
            self._reconstruction_slide(sample, sample_low, trainer),
        ]

    @staticmethod
    def _title_slide() -> Image.Image:
        """Render the presentation title slide."""
        fig = new_fig()
        fig.text(
            0.5,
            0.62,
            "MNIST Rate Coding with snnTorch",
            ha="center",
            fontsize=24,
            weight="bold",
        )
        fig.text(
            0.5,
            0.38,
            "A short visual tour of spike encoding",
            ha="center",
            fontsize=14,
        )
        return fig_to_image(fig)

    @staticmethod
    def _raster_slide(sample_low: Any, trainer: Any) -> Image.Image:
        """Render the input-layer raster slide."""
        fig = new_fig()
        ax = fig.add_subplot(111)
        splt.raster(
            sample_low.reshape(len(sample_low), -1), ax, s=1.5, c="black"
        )
        ax.set_title(f"Raster Plot - Input Layer (Gain = {trainer.gain})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")
        return fig_to_image(fig)

    @staticmethod
    def _reconstruction_slide(
        sample: Any, sample_low: Any, trainer: Any
    ) -> Image.Image:
        """Render the gain=1 vs low-gain reconstruction slide."""
        fig = new_fig()
        PresentationGifExporter._recon_panel(
            fig,
            121,
            time_averaged_image(sample, trainer.data_size),
            "Gain = 1",
        )
        PresentationGifExporter._recon_panel(
            fig,
            122,
            time_averaged_image(sample_low, trainer.data_size),
            f"Gain = {trainer.gain}",
        )
        fig.suptitle("Reconstruction (average spiking over time)")
        return fig_to_image(fig)

    @staticmethod
    def _recon_panel(fig: Any, index: int, image: Any, title: str) -> None:
        """Draw one reconstruction image panel."""
        ax = fig.add_subplot(index)
        ax.imshow(image, cmap="binary")
        ax.axis("off")
        ax.set_title(title)

    @staticmethod
    def _animation_frames(
        sample: Any, interval: int
    ) -> Tuple[List[Image.Image], List[int]]:
        """Render one GIF frame per spike time step."""
        frames: List[Image.Image] = []
        durations: List[int] = []
        for step in range(len(sample)):
            frames.append(PresentationGifExporter._animation_frame(
                sample, step, len(sample)
            ))
            durations.append(interval)
        return frames, durations

    @staticmethod
    def _animation_frame(sample: Any, step: int, total: int) -> Image.Image:
        """Render a single plasma-coloured spike frame."""
        fig = new_fig()
        ax = fig.add_subplot(111)
        ax.imshow(
            sample[step].cpu(),
            cmap="plasma",
            vmin=sample.min().item(),
            vmax=sample.max().item(),
        )
        ax.set_axis_off()
        fig.suptitle(
            f"Rate-coded spiking (Gain = 1)\n"
            f"Time step {step + 1} of {total}"
        )
        return fig_to_image(fig)
