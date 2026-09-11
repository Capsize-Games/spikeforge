"""Batch sparse event samples into the simulator's spike contract.

:class:`~snn_interpreter.events.event_bridge.EventSpikeBridge` densifies one
:class:`~snn_interpreter.events.event_sample.EventSample` at batch size 1 and
lays it out for a topology. This module stacks a chunk of samples along the
batch axis so an event epoch is the exact ``[T, B, F]``/``[T, B, C, H, W]``
shape the shared simulator and loss loop already consume. No coding math is
re-implemented here; the bridge remains the single conversion path.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from snn_interpreter.events.event_bridge import EventSpikeBridge
from snn_interpreter.events.event_source import EventSampleSource
from snn_interpreter.topology.spec import TopologySpec

#: Nominal batches one event epoch runs before the subset divisor applies.
EPOCH_BATCHES = 10

Batch = Tuple[torch.Tensor, torch.Tensor]


def batch_count(subset: int) -> int:
    """Return how many batches one event epoch visits for ``subset``."""
    return max(1, EPOCH_BATCHES // max(1, int(subset)))


def _one(
    source: EventSampleSource,
    index: int,
    spec: TopologySpec,
    bridge: EventSpikeBridge,
) -> Tuple[torch.Tensor, int]:
    """Return one bridged ``(spikes_without_batch, label)`` pair."""
    sample, label = source.load(source.clamp(index))
    spikes, _meta = bridge.encode(sample, spec)
    return spikes[:, 0], int(label)


def batch_event_samples(
    source: EventSampleSource,
    spec: TopologySpec,
    indices: Sequence[int],
    bridge: Optional[EventSpikeBridge] = None,
) -> Batch:
    """Stack ``indices`` of an event source into one ``[T, B, ...]`` batch."""
    encoder = bridge or EventSpikeBridge()
    pairs = [_one(source, index, spec, encoder) for index in indices]
    spikes = torch.stack([pair[0] for pair in pairs], dim=1)
    labels = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return spikes, labels


def event_batches(
    source: EventSampleSource,
    spec: TopologySpec,
    subset: int,
    batch_size: int,
) -> List[Batch]:
    """Return the epoch's ``(spikes, labels)`` batches in bridge layout."""
    size = max(1, int(batch_size))
    total = batch_count(subset) * size
    return [
        batch_event_samples(
            source, spec, range(start, min(start + size, total))
        )
        for start in range(0, total, size)
    ]
