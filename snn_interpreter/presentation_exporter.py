"""Export a combined looping presentation GIF of all visuals."""

import snntorch.spikeplot as splt

from snn_interpreter.exporter import Exporter
from snn_interpreter.plot_utils import (
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

    def export(self, filepath=None, slide_duration_ms=None):
        """Save the full visual tour as an infinitely looping GIF."""
        if filepath is None:
            filepath = self.DEFAULT_FILENAME
        if slide_duration_ms is None:
            slide_duration_ms = self.SLIDE_MS
        frames, durations = self._assemble(
            slide_duration_ms=slide_duration_ms
        )
        save_gif(filepath, frames, durations)

    def _assemble(self, slide_duration_ms):
        """Return (frames, durations) for every slide and animation."""
        trainer = self.trainer
        sample = spike_sample(trainer.spike_data)
        sample_low = spike_sample(trainer.spike_data_low_gain)

        frames, durations = [], []
        for slide in self._slides(
            sample, sample_low, trainer, slide_duration_ms
        ):
            frames.extend(slide.frames)
            durations.extend(slide.durations)
        anim_frames, anim_durations = self._animation_frames(
            sample, trainer.interval
        )
        frames.extend(anim_frames)
        durations.extend(anim_durations)
        return frames, durations

    def _slides(self, sample, sample_low, trainer, slide_ms):
        return [
            self._title_slide(slide_ms),
            self._raster_slide(sample_low, trainer, slide_ms),
            self._reconstruction_slide(
                sample, sample_low, trainer, slide_ms
            ),
        ]

    @staticmethod
    def _title_slide(slide_ms):
        fig = new_fig()
        fig.text(
            0.5, 0.62, "MNIST Rate Coding with snnTorch",
            ha="center", fontsize=24, weight="bold",
        )
        fig.text(
            0.5, 0.38, "A short visual tour of spike encoding",
            ha="center", fontsize=14,
        )
        return _Slide(fig_to_image(fig), slide_ms)

    @staticmethod
    def _raster_slide(sample_low, trainer, slide_ms):
        fig = new_fig()
        ax = fig.add_subplot(111)
        splt.raster(
            sample_low.reshape(len(sample_low), -1), ax, s=1.5, c="black"
        )
        ax.set_title(f"Raster Plot - Input Layer (Gain = {trainer.gain})")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Neuron Number")
        return _Slide(fig_to_image(fig), slide_ms)

    @staticmethod
    def _reconstruction_slide(sample, sample_low, trainer, slide_ms):
        fig = new_fig()
        PresentationGifExporter._recon_panel(
            fig, 121, time_averaged_image(sample, trainer.data_size),
            "Gain = 1",
        )
        PresentationGifExporter._recon_panel(
            fig, 122,
            time_averaged_image(sample_low, trainer.data_size),
            f"Gain = {trainer.gain}",
        )
        fig.suptitle("Reconstruction (average spiking over time)")
        return _Slide(fig_to_image(fig), slide_ms)

    @staticmethod
    def _recon_panel(fig, index, image, title):
        ax = fig.add_subplot(index)
        ax.imshow(image, cmap="binary")
        ax.axis("off")
        ax.set_title(title)

    @staticmethod
    def _animation_frames(sample, interval):
        frames = []
        durations = []
        for step, frame in enumerate(sample):
            fig = new_fig()
            ax = fig.add_subplot(111)
            ax.imshow(
                frame.cpu(),
                cmap="plasma",
                vmin=sample.min().item(),
                vmax=sample.max().item(),
            )
            ax.set_axis_off()
            fig.suptitle(
                f"Rate-coded spiking (Gain = 1)\n"
                f"Time step {step + 1} of {len(sample)}"
            )
            frames.append(fig_to_image(fig))
            durations.append(interval)
        return frames, durations


class _Slide:
    """One presentation slide rendered into reusable GIF frames."""

    def __init__(self, image, duration_ms):
        self.frames = [image]
        self.durations = [duration_ms]
