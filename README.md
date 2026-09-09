# snn-interpreter

A rate-coding demonstration for spiking neural networks (SNNs), built on
[snnTorch](https://snntorch.readthedocs.io/) and PyTorch. It loads a subset of
MNIST, converts the samples into rate-coded spike trains, and exports a set of
visual artifacts (MP4, looping GIFs, raster plots, and reconstructed images)
into the `build/` directory.

The encoding pipeline mirrors the concepts from the
[snnTorch tutorial on rate coding](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_1.html).

## Features

- Loads MNIST and reduces it with `snntorch.utils.data_subset`
- Encodes a mini-batch into spike trains with `spikegen.rate`
  (once at gain 1, once at a configurable lower gain)
- Exports:
  - An **MP4** animation of the rate-coded spikes
  - A **looping GIF** of the same animation (post-friendly)
  - A **reconstruction figure** comparing gain=1 vs. low-gain input
  - A **raster plot** of the input layer and a single spiking neuron
  - A combined **presentation GIF** (title, raster, reconstruction, animation)

## Requirements

- Python 3.8+
- `torch`, `torchvision`
- `snntorch`
- `matplotlib`, `Pillow`, `numpy`
- `ffmpeg` (only if the MP4 writer requires it)

Install the package and its dependencies:

```bash
pip install -e .
```

## Usage

Run the full pipeline from the project root:

```bash
python main.py
```

This constructs an `SNNTrainerLogger`, rate-codes the first mini-batch, and
writes every artifact under `build/`:

| File | Description |
| --- | --- |
| `build/spike_mnist_test.mp4` | Rate-coded spike animation (gain = 1) |
| `build/spike_mnist_test.gif` | Same animation as an infinite-loop GIF |
| `build/spike_reconstruction.png` | Reconstructed input, gain 1 vs. low gain |
| `build/spike_raster.png` | Input-layer + single-neuron raster plots |
| `build/spike_presentation.gif` | Combined looping presentation GIF |

`build/` is created automatically and is git-ignored, along with generated
media files, so build outputs are never committed.

### Programmatic use

```python
from snn_interpreter.logger import SNNTrainerLogger
from snn_interpreter.raster_exporter import RasterExporter
from snn_interpreter.presentation_exporter import PresentationGifExporter

trainer = SNNTrainerLogger(
    subset=10,
    vectorization_num_steps=10,
    vector_value=0.5,
    reconstruction_gain=0.25,
    animation_interval=100,
)
RasterExporter(trainer).export()
PresentationGifExporter(trainer).export()
```

Each exporter writes to `build/<DEFAULT_FILENAME>` unless a `filepath` is
passed explicitly.

## Project layout

```
main.py                     Thin entry point wiring trainer -> exporters
setup.py                    Packaging metadata (editable install)
snn_interpreter/
  trainer.py                SSNTrainer: MNIST loading + rate coding
  logger.py                 SNNTrainerLogger: logging subclass
  exporter.py               Exporter base + build/ output resolution
  plot_utils.py             Shared figure/GIF rendering helpers
  video_exporter.py         VideoExporter      -> build/*.mp4
  spike_gif_exporter.py     SpikeGifExporter   -> build/*.gif
  reconstruction_exporter.py ReconstructionExporter -> build/*.png
  raster_exporter.py        RasterExporter     -> build/*.png
  presentation_exporter.py  PresentationGifExporter -> build/*.gif
```

Code is kept tidy by construction: each file is under 200 lines, each class
lives in its own file, and every function stays under 20 lines.

## Notes

- The Bernoulli encoder in `spikegen.rate` is stochastic, so the reported
  "percent of time spiking" and the raster/spike patterns vary run to run;
  this is expected.
- Larger `vectorization_num_steps` values (e.g., 100) produce longer, richer
  spike animations and are easy to experiment with via the trainer argument.
