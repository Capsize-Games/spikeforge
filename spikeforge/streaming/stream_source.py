"""P1: a deterministic, anomaly-capable synthetic streaming dataset.

The generator builds a continuous multi-channel numeric stream (machine
telemetry, vibration, grid/sensor channels) as a concatenation of
class-labelled segments, injects transient and level-shift anomalies, and then
windows the stream with the frozen :class:`~spikeforge.streaming.window_spec.\
WindowSpec`. Labels are the class of a window's centre sample; the anomaly flag
is raised when any injected anomaly falls inside the window.

Everything is seeded, so a run is reproducible: the same
:class:`StreamSpec` yields the same stream, the same windows, and the same
labels. Train, validation, and test come from *different* seeds and the z-score
statistics are fitted on the train stream only, so no test information leaks
into the frozen windowing contract.

The module is pure torch and stdlib.
"""

from typing import Tuple

import torch

from spikeforge.streaming.stream_dataset import StreamDataset
from spikeforge.streaming.stream_generation import generate_stream
from spikeforge.streaming.stream_spec import StreamSpec
from spikeforge.streaming.stream_splits import StreamSplits
from spikeforge.streaming.window_spec import WindowSpec, fit_window_spec

#: Seed offsets that keep the three splits independent but reproducible.
_VAL_SEED_OFFSET = 101
_TEST_SEED_OFFSET = 202

__all__ = [
    "StreamDataset",
    "StreamSpec",
    "StreamSplits",
    "StreamTensors",
    "build_datasets",
    "generate_stream",
]


def _empty_dataset(windows: torch.Tensor) -> StreamDataset:
    """Return an empty windowed dataset with the right dtypes."""
    empty_labels = torch.zeros(0, dtype=torch.long)
    empty_flags = torch.zeros(0, dtype=torch.bool)
    return StreamDataset(windows, empty_labels, empty_flags)


def _window_dataset(
    spec: StreamSpec,
    stream: torch.Tensor,
    labels: torch.Tensor,
    anomaly: torch.Tensor,
    window_spec: WindowSpec,
) -> StreamDataset:
    """Window ``stream`` and attach the centre label and any-anomaly flag."""
    windows = window_spec.windows(stream)
    count = int(windows.size(0))
    if count == 0:
        return _empty_dataset(windows)
    starts = torch.arange(count) * spec.stride
    centres = starts + spec.length // 2
    window_labels = labels[centres]
    offsets = torch.arange(spec.length)
    index = starts.unsqueeze(1) + offsets.unsqueeze(0)
    window_anomaly = anomaly[index].any(dim=1)
    return StreamDataset(windows, window_labels, window_anomaly)


#: (stream, labels, anomaly) tensors returned by :func:`generate_stream`.
StreamTensors = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]


def _generate_splits(
    spec: StreamSpec,
) -> Tuple[StreamTensors, StreamTensors, StreamTensors]:
    """Generate the raw train/val/test streams from independent seeds."""
    train = generate_stream(spec, spec.seed)
    val = generate_stream(spec, spec.seed + _VAL_SEED_OFFSET)
    test = generate_stream(spec, spec.seed + _TEST_SEED_OFFSET)
    return train, val, test


def build_datasets(spec: StreamSpec) -> StreamSplits:
    """Return the deterministic train/val/test splits for ``spec``.

    The z-score statistics are fitted on the train stream and then frozen, so
    validation and test windows are normalised with train-only statistics.
    """
    spec.validate()
    train, val, test = _generate_splits(spec)
    window_spec = fit_window_spec(
        train[0], spec.length, spec.stride, spec.channel_names
    )
    return StreamSplits(
        train=_window_dataset(spec, *train, window_spec),
        val=_window_dataset(spec, *val, window_spec),
        test=_window_dataset(spec, *test, window_spec),
        window_spec=window_spec,
    )
