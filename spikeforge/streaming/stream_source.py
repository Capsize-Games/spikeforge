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


def _class_waveform(
    spec: StreamSpec, label: int, steps: int, start: int
) -> torch.Tensor:
    """Return the ``[steps, D]`` clean waveform for one class segment.

    Each class has its own fundamental frequency, per-channel phase, and
    amplitude, plus a strong class-specific level bump on one channel, so the
    classes are clearly separable in a window while keeping a smooth,
    temporally-correlated shape a spiking trunk can integrate.
    """
    time = torch.arange(start, start + steps, dtype=torch.float32)
    channels = torch.arange(spec.channels, dtype=torch.float32)
    frequency = 0.03 + 0.02 * float(label)
    phase = 0.4 * (channels + 1.0) + 0.3 * float(label)
    amplitude = 0.8 + 0.3 * float(label)
    carrier = amplitude * torch.sin(
        2.0 * 3.141592653589793 * frequency * time.unsqueeze(1) + phase
    )
    offset = torch.zeros(spec.channels, dtype=torch.float32)
    offset[label % spec.channels] = 2.5
    return carrier + offset


def _inject_anomaly(
    segment: torch.Tensor, spec: StreamSpec, generator: torch.Generator
) -> torch.Tensor:
    """Return ``segment`` with one transient or level-shift anomaly.

    A transient is a one- or two-sample spike burst on a random channel; a
    level shift adds a constant offset to one channel from a random sample
    onwards. Both change the window's temporal signature, which is what the
    anomaly head is meant to flag.
    """
    steps = segment.size(0)
    if steps < 2:
        return segment
    channel = int(
        torch.randint(0, spec.channels, (1,), generator=generator).item()
    )
    position = int(
        torch.randint(0, steps, (1,), generator=generator).item()
    )
    result = segment.clone()
    if bool(torch.randint(0, 2, (1,), generator=generator).item()):
        width = min(2, steps - position)
        result[position:position + width, channel] += 3.0
    else:
        result[position:, channel] += 1.5
    return result


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
    segments = []
    labels = []
    anomalies = []
    clock = 0
    for index in range(spec.segments):
        label = index % spec.classes
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
        empty_labels = torch.zeros(0, dtype=torch.long)
        empty_flags = torch.zeros(0, dtype=torch.bool)
        return StreamDataset(windows, empty_labels, empty_flags)
    starts = torch.arange(count) * spec.stride
    centres = starts + spec.length // 2
    window_labels = labels[centres]
    offsets = torch.arange(spec.length)
    index = starts.unsqueeze(1) + offsets.unsqueeze(0)
    window_anomaly = anomaly[index].any(dim=1)
    return StreamDataset(windows, window_labels, window_anomaly)


def build_datasets(spec: StreamSpec) -> StreamSplits:
    """Return the deterministic train/val/test splits for ``spec``.

    The z-score statistics are fitted on the train stream and then frozen, so
    validation and test windows are normalised with train-only statistics.
    """
    spec.validate()
    train_stream, train_labels, train_anomaly = generate_stream(
        spec, spec.seed
    )
    window_spec = fit_window_spec(
        train_stream, spec.length, spec.stride, spec.channel_names
    )
    val_stream, val_labels, val_anomaly = generate_stream(
        spec, spec.seed + _VAL_SEED_OFFSET
    )
    test_stream, test_labels, test_anomaly = generate_stream(
        spec, spec.seed + _TEST_SEED_OFFSET
    )
    return StreamSplits(
        train=_window_dataset(
            spec, train_stream, train_labels, train_anomaly, window_spec
        ),
        val=_window_dataset(
            spec, val_stream, val_labels, val_anomaly, window_spec
        ),
        test=_window_dataset(
            spec, test_stream, test_labels, test_anomaly, window_spec
        ),
        window_spec=window_spec,
    )
