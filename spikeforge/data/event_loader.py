"""Load event datasets through the optional ``tonic`` package.

Importing ``tonic`` is confined to :mod:`spikeforge.events.tonic_api`;
this loader reaches the package only through that probe, so it imports
safely whether or not the ``events`` extra is installed.

Tonic event tensors use a structured ``(x, y, t, p)`` layout where ``x`` is
the column, ``y`` the row, ``t`` a microsecond timestamp, and ``p`` a
boolean polarity (``True`` is ON). We map that to
:class:`~spikeforge.events.event_sample.EventSample` by keeping ``x``
and ``y``, converting ``p`` to ``+1`` (ON) / ``-1`` (OFF), and binning each
microsecond timestamp into ``num_steps`` uniform 0-based bins spanning the
sample's own ``[t_min, t_max]`` range. The ``(H, W)`` sensor layout comes
from tonic's ``sensor_size`` ``(width, height, channels)`` triple.

Downloads run through the existing isolated worker
(``spikeforge.data.download_cli``): the child process constructs the
tonic dataset, whose own cache writes under ``DATA_DIR/events`` so the
server's directory-size progress poll and its cancel/terminate path keep
working unchanged. Tonic has no ``download=False`` switch, so
:func:`ensure_event_dataset` constructs the dataset and lets tonic skip the
network when the cache is warm; the loader never touches the network at
import time.

Every entry point takes a ``split``. The split is not a boolean here: the
registry records the constructor arguments that select it, because
``NMNIST`` and ``DVSGesture`` take ``train=True/False`` while ``SSC`` takes
``split="train"/"test"``. Both splits of one dataset share a cache root, so
tonic's own ``train``/``test`` subdirectories keep the progress poll working.
"""

import os
from typing import Any, Optional, Tuple

import numpy as np
import torch

from spikeforge.config import DATA_DIR
from spikeforge.data.dataset_spec import DatasetSpec
from spikeforge.data.datasets import dataset_spec, dataset_split_kwargs
from spikeforge.data.event_errors import EventsExtraMissingError
from spikeforge.events import tonic_api
from spikeforge.events.event_sample import EventSample

#: Default number of time bins a loaded sample is binned into.
DEFAULT_NUM_STEPS = 10
#: The split loaded when a caller names none.
DEFAULT_SPLIT = "train"


def _root(spec: DatasetSpec, save_to: Optional[str]) -> str:
    """Return the on-disk cache directory for an event dataset."""
    if save_to is not None:
        return save_to
    return os.path.join(DATA_DIR, "events", spec.name)


def _dataset(
    name: str, save_to: Optional[str], split: str = DEFAULT_SPLIT
) -> Any:
    """Instantiate one split of the tonic dataset named by a registry key.

    The split's constructor arguments come from the registry rather than
    from this module, because tonic spells them differently per dataset;
    asking for a split a dataset does not declare raises
    :class:`~spikeforge.data.event_errors.EventSplitMissingError`.
    """
    spec = dataset_spec(name)
    if spec.modality != "event":
        raise ValueError(f"dataset {spec.name!r} is not an event dataset")
    kwargs = dataset_split_kwargs(spec.name, split)
    cls = tonic_api.dataset_class(spec.tonic_class or "")
    if cls is None:
        raise EventsExtraMissingError(spec.name)
    return cls(_root(spec, save_to), **kwargs)


def _shape(dataset: Any) -> Tuple[int, int]:
    """Return the ``(H, W)`` sensor layout from tonic's ``sensor_size``."""
    width, height, _channels = dataset.sensor_size
    return int(height), int(width)


def _field(events: Any, name: str) -> torch.Tensor:
    """Return one flat integer column from a structured event stream."""
    try:
        column = events[name]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError(f"event stream has no {name!r} field") from exc
    return torch.as_tensor(np.ascontiguousarray(column)).reshape(-1).long()


def _polarity(p: torch.Tensor) -> torch.Tensor:
    """Map tonic's 1/0 ON/OFF polarity to our ``+1``/``-1`` convention."""
    return torch.where(p > 0, 1, -1).long()


def _time_bins(t: torch.Tensor, num_steps: int) -> torch.Tensor:
    """Bin microsecond timestamps into ``num_steps`` 0-based bins."""
    steps = max(1, int(num_steps))
    if t.numel() == 0:
        return t
    low = int(t.min())
    span = int(t.max()) - low
    if span <= 0:
        return torch.zeros_like(t)
    scaled = (t - low).float() / float(span)
    return (scaled * steps).floor().long().clamp(0, steps - 1)


def events_to_sample(
    events: Any, shape: Tuple[int, int], num_steps: int
) -> EventSample:
    """Convert a tonic ``(x, y, t, p)`` stream to an ``EventSample``."""
    x = _field(events, "x")
    y = _field(events, "y")
    t = _time_bins(_field(events, "t"), num_steps)
    p = _polarity(_field(events, "p"))
    return EventSample(x, y, t, p, shape, num_steps)


def load_event_pair(
    name: str,
    index: int = 0,
    num_steps: int = DEFAULT_NUM_STEPS,
    save_to: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
) -> Tuple[EventSample, int]:
    """Load sample ``index`` of ``split`` as an ``(EventSample, label)`` pair.

    Raises :class:`EventsExtraMissingError` when the ``events`` extra is
    not installed, and
    :class:`~spikeforge.data.event_errors.EventSplitMissingError` when the
    dataset declares no such split.
    """
    dataset = _dataset(name, save_to, split)
    events, target = dataset[int(index)]
    sample = events_to_sample(events, _shape(dataset), num_steps)
    return sample, int(target)


def load_event_sample(
    name: str,
    index: int = 0,
    num_steps: int = DEFAULT_NUM_STEPS,
    save_to: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
) -> EventSample:
    """Load sample ``index`` of ``split`` as a binned :class:`EventSample`."""
    return load_event_pair(name, index, num_steps, save_to, split)[0]


def ensure_event_dataset(
    name: str, save_to: Optional[str] = None, split: str = DEFAULT_SPLIT
) -> None:
    """Warm one split of an event dataset's cache without loading a sample.

    Constructing the tonic dataset triggers its cached download; the
    isolated worker calls this once per declared split so no network I/O
    runs inside the server, including the held-out fetch that scoring
    would otherwise trigger mid-training.
    """
    _dataset(name, save_to, split)
