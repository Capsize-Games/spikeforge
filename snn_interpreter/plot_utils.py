"""Shared figure and GIF helpers used by the output exporters."""

import io

import matplotlib.pyplot as plt
from PIL import Image


def spike_sample(spike_data):
    """First MNIST sample with the batch dim removed."""
    return spike_data[:, 0, 0]


def time_averaged_image(sample, data_size):
    """Average spiking over time, reshaped back into an image."""
    return sample.mean(axis=0).reshape(data_size).cpu()


def neuron_index(sample):
    """Index of the neuron that spikes most often."""
    flat = sample.reshape(sample.size(0), -1)
    return int(flat.sum(dim=0).argmax().item())


def new_fig(width=8, height=5, dpi=100):
    """Create a white-background figure sized for a slide."""
    return plt.figure(facecolor="w", figsize=(width, height), dpi=dpi)


def fig_to_image(fig):
    """Render a figure to a PIL RGB image and close the figure."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB").copy()


def save_gif(filepath, frames, durations):
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
