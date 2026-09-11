"""Export the rate-coded spike train as a looping GIF."""

from typing import Optional

import torch
from PIL import Image

from snn_interpreter.exporters.exporter import Exporter
from snn_interpreter.exporters.plot_utils import (
    fig_to_image,
    new_fig,
    save_gif,
    spike_sample,
)


class SpikeGifExporter(Exporter):
    """Render each time step as a plasma image in an infinite GIF."""

    DEFAULT_FILENAME = "spike_mnist_test.gif"

    def export(self, filepath: Optional[str] = None) -> None:
        """Save the spike animation as an infinitely looping GIF."""
        filepath = filepath or self.output_path
        sample = spike_sample(self.trainer.spike_data)
        frames = [self._frame(sample, step) for step in range(len(sample))]
        durations = [self.trainer.interval] * len(frames)
        save_gif(filepath, frames, durations)

    def _frame(self, sample: torch.Tensor, step: int) -> Image.Image:
        """Render one time step as a plasma-coloured PIL frame."""
        fig = new_fig()
        ax = fig.add_subplot(111)
        ax.imshow(
            sample[step].cpu(),
            cmap="plasma",
            vmin=sample.min().item(),
            vmax=sample.max().item(),
        )
        ax.set_axis_off()
        return fig_to_image(fig)
