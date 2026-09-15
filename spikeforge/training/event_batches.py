"""Batch sparse event samples into the simulator's spike contract.

:class:`~spikeforge.events.event_bridge.EventSpikeBridge` densifies one
:class:`~spikeforge.events.event_sample.EventSample` at batch size 1 and
lays it out for a topology. This module stacks a chunk of samples along the
batch axis so an event epoch is the exact ``[T, B, F]``/``[T, B, C, H, W]``
shape the shared simulator and loss loop already consume. No coding math is
re-implemented here; the bridge remains the single conversion path.
"""

from typing import List, Optional, Sequence, Tuple

import torch

from spikeforge.events.event_bridge import EventSpikeBridge
from spikeforge.events.event_source import EventSampleSource
from spikeforge.topology.spec import TopologySpec

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


def visit_order(
    total: int, shuffle: bool, seed: Optional[int] = None
) -> List[int]:
    """Return the sample indices one epoch visits, in visiting order.

    Sequential unless ``shuffle``. Shuffling is not cosmetic here: real event
    datasets arrive grouped by class -- SSC's training split is 35 contiguous
    runs over 75,466 samples -- so reading them in order makes every batch a
    single class and training collapses to predicting whichever class it is
    currently being shown. ``seed`` makes the permutation reproducible, which a
    published row needs.
    """
    if not shuffle:
        return list(range(total))
    generator = None
    if seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return [int(i) for i in torch.randperm(total, generator=generator)]


def event_batches(
    source: EventSampleSource,
    spec: TopologySpec,
    subset: int,
    batch_size: int,
    samples: Optional[int] = None,
    shuffle: bool = False,
    seed: Optional[int] = None,
) -> List[Batch]:
    """Return the epoch's ``(spikes, labels)`` batches in bridge layout.

    By default an epoch is :func:`batch_count` batches, which keeps the live
    dashboard responsive: it is a demo of the pipeline, not a full pass. That
    default is a cap, not a fraction — with ``subset=1`` it still visits only
    ``EPOCH_BATCHES * batch_size`` samples however large the split is, so it
    must not be what a published number is trained on.

    ``samples`` names the epoch's length outright. Pass the split's own size
    (``source.size()``) to train on all of it, which is what a reference
    checkpoint claiming the full training split requires. ``subset`` is
    bypassed when ``samples`` is given, since the two would otherwise both be
    trying to set the same thing.

    ``shuffle`` visits the samples in a seeded random order, which a training
    epoch over a class-ordered dataset requires; see :func:`visit_order`. It
    matches the image path, whose loader is built with ``shuffle=train``.
    """
    size = max(1, int(batch_size))
    total = (
        max(1, int(samples))
        if samples is not None
        else batch_count(subset) * size
    )
    order = visit_order(total, shuffle, seed)
    return [
        batch_event_samples(source, spec, order[start:start + size])
        for start in range(0, total, size)
    ]
