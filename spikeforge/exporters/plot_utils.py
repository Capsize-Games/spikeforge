"""Shared figure and GIF helpers used by the output exporters."""

import io
from typing import List, Tuple

import matplotlib.pyplot as plt
import torch
from PIL import Image


def spike_sample(spike_data: torch.Tensor) -> torch.Tensor:
    """First MNIST sample with the batch dim removed."""
    return spike_data[:, 0, 0]


def time_averaged_image(
    sample: torch.Tensor, data_size: Tuple[int, int]
) -> torch.Tensor:
    """Average spiking over time, reshaped back into an image."""
    return sample.mean(axis=0).reshape(data_size).cpu()


def neuron_index(sample: torch.Tensor) -> int:
    """Index of the neuron that spikes most often."""
    flat = sample.reshape(sample.size(0), -1)
    return int(flat.sum(dim=0).argmax().item())


def new_fig(width: float = 8, height: float = 5, dpi: int = 100) -> plt.Figure:
    """Create a white-background figure sized for a slide."""
    return plt.figure(facecolor="w", figsize=(width, height), dpi=dpi)


def fig_to_image(fig: plt.Figure) -> Image.Image:
    """Render a figure to a PIL RGB image and close the figure."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def save_gif(
    filepath: str, frames: List[Image.Image], durations: List[int]
) -> None:
    """Persist equally sized PIL frames as an infinitely looping GIF."""
    target = frames[0].size
    frames = [
        frame.resize(target) if frame.size != target else frame
        for frame in frames
    ]
    frames[0].save(
        filepath,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
