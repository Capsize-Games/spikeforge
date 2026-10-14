"""Generate the seeded multi-channel stream behind the windowed dataset."""

from typing import Optional, Tuple

import torch

from spikeforge.streaming.stream_spec import StreamSpec


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
