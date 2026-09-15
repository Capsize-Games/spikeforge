"""Load event datasets through the optional ``tonic`` package.

Importing ``tonic`` is confined to :mod:`spikeforge.events.tonic_api`;
this loader reaches the package only through that probe, so it imports
safely whether or not the ``events`` extra is installed.

Tonic event tensors use a structured ``(x, y, t, p)`` layout where ``x`` is
the column, ``y`` the row, ``t`` a microsecond timestamp, and ``p`` a
boolean polarity (``True`` is ON). Auditory sensors are the exception: a
cochlea has channels, not pixel rows, so tonic's SHD/SSC streams are
``(t, x, p)`` on a sensor declared one row tall and every event sits on row
0. We map either shape to
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
from spikeforge.data.event_errors import (
    EventsExtraMissingError,
    EventTimestampError,
)
from spikeforge.events import hsd_reader, tonic_api
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


def open_event_dataset(
    name: str, save_to: Optional[str] = None, split: str = DEFAULT_SPLIT
) -> Any:
    """Instantiate one split of the tonic dataset named by a registry key.

    Constructing a tonic dataset is not free -- it indexes the split's files
    -- and it is also what triggers the cached download. So a caller that
    reads many samples should open the dataset **once** and hold it (see
    :class:`~spikeforge.events.event_source.EventSampleSource`) rather than
    calling :func:`load_event_pair` per sample.

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
    dataset = cls(_root(spec, save_to), **kwargs)
    if spec.native_reader == hsd_reader.HSD:
        # Tonic did the download and owns the cache layout; only its
        # per-sample decode is replaced. See `hsd_reader` for why.
        return hsd_reader.open_split(dataset)
    return dataset


def sample_from(
    dataset: Any, index: int, num_steps: int = DEFAULT_NUM_STEPS
) -> Tuple[EventSample, int]:
    """Return one ``(EventSample, label)`` pair from an already-open dataset.

    Split from :func:`open_event_dataset` so a caller holding the dataset
    pays the indexing cost once instead of once per sample.
    """
    events, target = dataset[int(index)]
    sample = events_to_sample(events, _shape(dataset), num_steps)
    return sample, int(target)


def _shape(dataset: Any) -> Tuple[int, int]:
    """Return the ``(H, W)`` sensor layout from tonic's ``sensor_size``."""
    width, height, _channels = dataset.sensor_size
    return int(height), int(width)


def _field(events: Any, name: str) -> torch.Tensor:
    """Return one flat integer column from a structured event stream."""
    try:
        column = events[name]
    # A mapping raises KeyError for an absent key; a numpy structured array
    # raises ValueError. Both mean the same thing and both get the named
    # message rather than leaking the container's own wording.
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"event stream has no {name!r} field") from exc
    return torch.as_tensor(np.ascontiguousarray(column)).reshape(-1).long()


def _has_field(events: Any, name: str) -> bool:
    """Return True when a structured event stream carries ``name``."""
    try:
        events[name]
    except (KeyError, IndexError, TypeError, ValueError):
        return False
    return True


def _rows(events: Any, x: torch.Tensor, height: int) -> torch.Tensor:
    """Return each event's row index, or zeros for a one-row sensor.

    Auditory sensors have no second spatial axis. Tonic's SHD and SSC declare
    ``sensor_size = (700, 1, 1)`` and their streams carry ``(t, x, p)`` with
    no ``y`` field at all, because a cochlea channel is not a pixel row. On a
    sensor whose declared height is 1 every event therefore sits on row 0.

    A missing ``y`` on a taller sensor is still an error: there the field is
    absent *data* rather than an axis the recording does not have, and
    defaulting it would silently collapse a 2-D recording onto one row.
    """
    if height == 1 and not _has_field(events, "y"):
        return torch.zeros_like(x)
    return _field(events, "y")


def _polarity(p: torch.Tensor) -> torch.Tensor:
    """Map tonic's 1/0 ON/OFF polarity to our ``+1``/``-1`` convention."""
    return torch.where(p > 0, 1, -1).long()


def _checked_times(t: torch.Tensor) -> torch.Tensor:
    """Return ``t`` unchanged, or refuse a stream whose times are impossible.

    Binning is deliberately forgiving -- a zero-width span is legitimate for
    a sample whose events share one timestamp -- so it cannot tell a real
    instant from timestamps that arrived destroyed. This check runs before
    binning, where the difference is still visible.
    """
    if t.numel() and int(t.min()) < 0:
        raise EventTimestampError(f"(minimum {int(t.min())})")
    return t


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
    """Convert a tonic ``(x, y, t, p)`` stream to an ``EventSample``.

    ``y`` is optional on a sensor one row tall; see :func:`_rows`.

    Raises :class:`~spikeforge.data.event_errors.EventTimestampError` when the
    timestamps cannot be recording times, rather than binning garbage into a
    tensor that trains without complaint.
    """
    x = _field(events, "x")
    y = _rows(events, x, shape[0])
    t = _checked_times(_field(events, "t"))
    p = _polarity(_field(events, "p"))
    return EventSample(x, y, _time_bins(t, num_steps), p, shape, num_steps)


def load_event_pair(
    name: str,
    index: int = 0,
    num_steps: int = DEFAULT_NUM_STEPS,
    save_to: Optional[str] = None,
    split: str = DEFAULT_SPLIT,
) -> Tuple[EventSample, int]:
    """Load sample ``index`` of ``split`` as an ``(EventSample, label)`` pair.

    Opens the dataset and reads one sample, so it is the convenient form for
    a one-off read and the wrong one for a loop: reading ``n`` samples this
    way indexes the split ``n`` times. Hold an
    :class:`~spikeforge.events.event_source.EventSampleSource` for that.

    Raises :class:`EventsExtraMissingError` when the ``events`` extra is
    not installed, and
    :class:`~spikeforge.data.event_errors.EventSplitMissingError` when the
    dataset declares no such split.
    """
    return sample_from(
        open_event_dataset(name, save_to, split), index, num_steps
    )


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
    open_event_dataset(name, save_to, split)
