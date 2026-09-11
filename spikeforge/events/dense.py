"""Dense accumulations of a sparse :class:`EventSample`.

Two documented conventions are exposed, both time-major ``[T, ...]``:

* :func:`to_frames` returns the binary frame stack ``[T, 2, H, W]`` where
  channel ``0`` is the ON count and channel ``1`` the OFF count, each entry
  binarised to ``0.0``/``1.0``.
* :func:`to_voxel` returns the spike-count/voxel form ``[T, 2, H, W]`` whose
  entries are the number of events that fired in that bin, channel, row,
  and column (the "spike rate" before any normalisation).

Zero events in a bin leave every entry at ``0``.
"""

import torch

from spikeforge.events.event_sample import EventSample


def _channel(polarity: torch.Tensor) -> torch.Tensor:
    """Return the ON/OFF channel index: ``0`` for ``+1``, ``1`` for ``-1``."""
    return (polarity < 0).long()


def _flat_index(sample: EventSample) -> torch.Tensor:
    """Return the flattened channel-major index of every event."""
    height, width = sample.shape
    stride = height * width
    channel = _channel(sample.p) * stride
    positional = sample.y * width + sample.x
    return sample.t * (2 * stride) + channel + positional


def to_voxel(sample: EventSample) -> torch.Tensor:
    """Accumulate events into the ``[T, 2, H, W]`` spike-count voxel.

    Channel ``0`` counts ON events, channel ``1`` counts OFF events, and
    duplicate events in the same bin sum rather than clip.
    """
    height, width = sample.shape
    voxel = torch.zeros(sample.num_steps, 2, height, width)
    index = _flat_index(sample)
    ones = torch.ones(index.numel())
    voxel.reshape(-1).index_add_(0, index, ones)
    return voxel


def to_frames(sample: EventSample) -> torch.Tensor:
    """Return the binary frame stack ``[T, 2, H, W]`` for ``sample``.

    Every entry is ``1.0`` when at least one event of that polarity fired
    in the bin and ``0.0`` otherwise, matching the simulator's binary spike
    convention without losing the ON/OFF split.
    """
    return (to_voxel(sample) > 0).to(torch.float32)
