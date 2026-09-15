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

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from spikeforge.streaming.window_spec import WindowSpec, fit_window_spec

#: Seed offsets that keep the three splits independent but reproducible.
_VAL_SEED_OFFSET = 101
_TEST_SEED_OFFSET = 202


@dataclass(frozen=True)
class StreamSpec:
    """The geometry and signal shape of one synthetic stream."""

    #: Samples per window (the ``L`` axis) and the window stride.
    length: int = 16
    stride: int = 8
    #: Number of numeric channels (the ``D`` axis).
    channels: int = 4
    #: Number of class labels the waveform generator separates.
    classes: int = 3
    #: Timesteps per class-labelled segment and how many segments to stitch.
    segment_length: int = 32
    segments: int = 12
    #: Probability that a segment carries an injected anomaly.
    anomaly_rate: float = 0.25
    #: Standard deviation of the additive Gaussian sensor noise.
    noise: float = 0.15
    #: Base seed; validation/test derive from it deterministically.
    seed: int = 0

    def validate(self) -> None:
        """Raise :class:`ValueError` when the spec is unusable."""
        if self.length < 1 or self.stride < 1:
            raise ValueError("length and stride must be >= 1")
        if self.channels < 1:
            raise ValueError("channels must be >= 1")
        if self.classes < 2:
            raise ValueError("classes must be >= 2")
        if self.segment_length < 1 or self.segments < 1:
            raise ValueError("segment_length and segments must be >= 1")
        if not 0.0 <= self.anomaly_rate <= 1.0:
            raise ValueError("anomaly_rate must be in [0, 1]")
        if self.noise < 0.0:
            raise ValueError("noise must be >= 0")

    @property
    def channel_names(self) -> Tuple[str, ...]:
        """Return the stable channel order ``c0, c1, ...``."""
        return tuple(f"c{index}" for index in range(self.channels))


def _carrier_wave(
    time: torch.Tensor, channels: torch.Tensor, label: int
) -> torch.Tensor:
    """Return the sinusoidal carrier for one class label."""
    frequency = 0.03 + 0.02 * float(label)
    phase = 0.4 * (channels + 1.0) + 0.3 * float(label)
    amplitude = 0.8 + 0.3 * float(label)
    return amplitude * torch.sin(
        2.0 * 3.141592653589793 * frequency * time.unsqueeze(1) + phase
    )


def _class_waveform(
    spec: StreamSpec, label: int, steps: int, start: int
) -> torch.Tensor:
    """Return the ``[steps, D]`` clean waveform for one class segment.

    Each class has its own frequency/phase/amplitude plus a level bump on
    one channel, so classes stay separable in a smooth, temporally
    correlated shape a spiking trunk can integrate.
    """
    time = torch.arange(start, start + steps, dtype=torch.float32)
    channels = torch.arange(spec.channels, dtype=torch.float32)
    carrier = _carrier_wave(time, channels, label)
    offset = torch.zeros(spec.channels, dtype=torch.float32)
    offset[label % spec.channels] = 2.5
    return carrier + offset


def _random_anomaly_site(
    spec: StreamSpec, steps: int, generator: torch.Generator
) -> Tuple[int, int]:
    """Return a random ``(channel, position)`` for an injected anomaly."""
    channel = int(
        torch.randint(0, spec.channels, (1,), generator=generator).item()
    )
    position = int(
        torch.randint(0, steps, (1,), generator=generator).item()
    )
    return channel, position


def _inject_anomaly(
    segment: torch.Tensor, spec: StreamSpec, generator: torch.Generator
) -> torch.Tensor:
    """Return ``segment`` with one transient or level-shift anomaly.

    Either a spike burst or a constant offset on a random channel, changing
    the temporal signature the anomaly head is meant to flag.
    """
    steps = segment.size(0)
    if steps < 2:
        return segment
    channel, position = _random_anomaly_site(spec, steps, generator)
    result = segment.clone()
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        width = min(2, steps - position)
        result[position:position + width, channel] += 3.0
    else:
        result[position:, channel] += 1.5
    return result


def _generate_segment(
    spec: StreamSpec, label: int, clock: int, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return one noisy, possibly anomalous ``[segment_length, D]`` segment."""
    clean = _class_waveform(spec, label, spec.segment_length, clock)
    noise = torch.randn(
        spec.segment_length, spec.channels, generator=generator
    )
    segment = clean + spec.noise * noise
    mask = torch.zeros(spec.segment_length, dtype=torch.bool)
    if spec.anomaly_rate > 0.0:
        roll = float(torch.rand(1, generator=generator).item())
        if roll < spec.anomaly_rate:
            segment = _inject_anomaly(segment, spec, generator)
            mask = torch.ones(spec.segment_length, dtype=torch.bool)
    return segment, mask


def _generate_segments(
    spec: StreamSpec, generator: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate and concatenate every class-labelled segment."""
    segments, labels, anomalies = [], [], []
    clock = 0
    for index in range(spec.segments):
        label = index % spec.classes
        segment, mask = _generate_segment(spec, label, clock, generator)
        segments.append(segment)
        labels.append(
            torch.full((spec.segment_length,), label, dtype=torch.long)
        )
        anomalies.append(mask)
        clock += spec.segment_length
    return (
        torch.cat(segments, dim=0),
        torch.cat(labels, dim=0),
        torch.cat(anomalies, dim=0),
    )


def generate_stream(
    spec: StreamSpec, seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(stream, labels, anomaly)`` for one split.

    ``stream`` is ``[N, D]`` float32, ``labels`` is ``[N]`` int64 (the class of
    every sample), and ``anomaly`` is ``[N]`` bool (the injected-anomaly mask).
    The class of segment ``i`` is ``i % classes`` so every class is present.
    """
    spec.validate()
    base_seed = spec.seed if seed is None else int(seed)
    generator = torch.Generator().manual_seed(int(base_seed))
    return _generate_segments(spec, generator)


@dataclass(frozen=True)
class StreamDataset:
    """Windowed, z-scored samples with a class label and an anomaly flag."""

    windows: torch.Tensor
    labels: torch.Tensor
    anomaly: torch.Tensor

    def __len__(self) -> int:
        """Return the number of windows."""
        return int(self.windows.size(0))


@dataclass(frozen=True)
class StreamSplits:
    """The train/validation/test split plus the frozen windowing contract."""

    train: StreamDataset
    val: StreamDataset
    test: StreamDataset
    window_spec: WindowSpec


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
