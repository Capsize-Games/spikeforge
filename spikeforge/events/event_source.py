"""Event-modality samples: real ``tonic`` datasets or explicit synthetic.

The loader itself lives in :mod:`spikeforge.data.event_loader`; this
source decides which backend serves a sample. It prefers the real dataset
whenever ``tonic`` can load it, and falls back to the deterministic
generators in :mod:`spikeforge.events.synthetic` only for the offline
path. The chosen backend is recorded in :attr:`origin` and
:attr:`description`, so a synthetic stream is never presented as a real
recording.

One source serves exactly one dataset split. Callers that need both — the
training engine, which scores a held-out set — hold two sources rather than
passing a flag down the load path, so the split is settled by the object
that opened the dataset. Under ``tonic`` the split is the real one the
recording ships. Under the synthetic backend there is no recording to
withhold, so the two splits are *generated*: each draws from its own
disjoint index window (see :data:`SYNTHETIC_SIZE` and
:data:`SYNTHETIC_TEST_SIZE`) and lays its dots in a different column band,
and :attr:`description` keeps saying so.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from spikeforge.data import event_loader
from spikeforge.data.datasets import (
    dataset_available,
    dataset_info,
    dataset_split_kwargs,
)
from spikeforge.events import synthetic
from spikeforge.events.event_sample import EventSample

#: Backend names reported by :attr:`EventSampleSource.origin`.
TONIC = "tonic"
SYNTHETIC = "synthetic"
#: The split a source serves when a caller names none.
DEFAULT_SPLIT = "train"
#: Square sensor every synthetic stream is generated on, so one sample can
#: drive both feature (28*28) and spatial (28x28) input stages.
SYNTHETIC_SHAPE: Tuple[int, int] = (28, 28)
#: Distinct synthetic samples the offline source serves before repeating.
SYNTHETIC_SIZE = 64
#: Distinct synthetic samples the held-out split serves. They are drawn from
#: indices ``[SYNTHETIC_SIZE, SYNTHETIC_SIZE + SYNTHETIC_TEST_SIZE)``, which
#: the training window never reaches.
SYNTHETIC_TEST_SIZE = 16
#: Time bins synthetic streams span.
SYNTHETIC_STEPS = 10
#: Columns the held-out synthetic split shifts its dots by. The generator's
#: index term spans six columns, so a shift this size puts the two splits'
#: starting positions in bands that cannot overlap — the training stream is
#: left untouched, and no generated test sample can coincide with a training
#: one.
_SYNTHETIC_TEST_SHIFT = 10
#: How each split is named in provenance strings.
_SPLIT_LABEL = {DEFAULT_SPLIT: "", "test": "held-out "}


class EventSampleSource:
    """Serve ``(EventSample, label)`` pairs for one split of a dataset.

    ``synthetic_only`` forces the offline generator even when ``tonic`` is
    installed, which keeps tests and demos network-free. Otherwise the real
    dataset is used whenever its loader is available.

    ``split`` is resolved against the registry at construction, so a
    dataset that declares no such split fails here with
    :class:`~spikeforge.data.event_errors.EventSplitMissingError` instead of
    quietly serving another one. The check runs on both backends, because
    what a dataset partitions into is a property of the dataset, not of
    whether ``tonic`` happens to be installed.
    """

    def __init__(
        self,
        dataset: str = "n_mnist",
        synthetic_only: bool = False,
        num_steps: int = SYNTHETIC_STEPS,
        save_to: Optional[str] = None,
        split: str = DEFAULT_SPLIT,
    ) -> None:
        """Resolve the split, choose the backend, cache the class count."""
        self._dataset = dataset
        self._num_steps = int(num_steps)
        self._save_to = save_to
        self._split = split
        # Resolved for the check, not the value: an undeclared split must
        # fail here, on either backend, not at the first held-out score.
        dataset_split_kwargs(dataset, split)
        self._classes = dataset_info(dataset)[0]
        self._origin = self._pick(synthetic_only)

    def _pick(self, synthetic_only: bool) -> str:
        """Return the backend name this source will use."""
        if synthetic_only or not dataset_available(self._dataset):
            return SYNTHETIC
        return TONIC

    def clamp(self, index: int) -> int:
        """Wrap an index into this split's valid range on the backend."""
        if self._origin == SYNTHETIC:
            base, size = self.synthetic_range()
            return base + int(index) % size
        return max(0, int(index))

    def synthetic_range(self) -> Tuple[int, int]:
        """Return the ``(base, size)`` index window this split generates.

        The training and held-out windows do not overlap, so a synthetic
        held-out sample is never one the training stream also served.
        """
        if self._split == DEFAULT_SPLIT:
            return 0, SYNTHETIC_SIZE
        return SYNTHETIC_SIZE, SYNTHETIC_TEST_SIZE

    def load(self, index: int) -> Tuple[EventSample, int]:
        """Return the ``(sample, label)`` pair for ``index`` in this split."""
        if self._origin == TONIC:
            return event_loader.load_event_pair(
                self._dataset, index, self._num_steps, self._save_to,
                self._split,
            )
        return self._synthetic(index), self._synthetic_label(index)

    def _synthetic(self, index: int) -> EventSample:
        """Return a deterministic two-polarity stream for ``index``.

        The held-out split shifts both dots' starting columns clear of the
        training split's band, so its samples differ from every training
        one even where the index terms repeat. Nothing here is withheld
        from a recording: both splits are generated, which
        :attr:`description` states.
        """
        shift = 0 if self._split == DEFAULT_SPLIT else _SYNTHETIC_TEST_SHIFT
        on = synthetic.moving_dot(
            num_steps=self._num_steps, shape=SYNTHETIC_SHAPE,
            start=(2.0 + index % 8, 2.0 + shift + index % 6),
            velocity=(0.8, 1.0), seed=index,
        )
        off = synthetic.moving_dot(
            num_steps=self._num_steps, shape=SYNTHETIC_SHAPE,
            start=(20.0 - index % 6, 20.0 - shift - index % 8),
            velocity=(-0.8, -1.0), polarity=-1, seed=index,
        )
        return EventSample(
            torch.cat([on.x, off.x]), torch.cat([on.y, off.y]),
            torch.cat([on.t, off.t]), torch.cat([on.p, off.p]),
            SYNTHETIC_SHAPE, self._num_steps,
        )

    def _synthetic_label(self, index: int) -> int:
        """Return a stable pseudo-label for a synthetic sample."""
        return int(index) % max(1, self._classes)

    def info(self) -> Dict[str, Any]:
        """Return JSON-able provenance metadata for the source."""
        return {
            "modality": "event",
            "origin": self._origin,
            "dataset": self._dataset,
            "split": self._split,
            "description": self.description,
        }

    @property
    def description(self) -> str:
        """Return a one-line, human-readable provenance description."""
        label = _SPLIT_LABEL.get(self._split, f"{self._split} ")
        if self._origin == TONIC:
            return f"{self._dataset} {label}events loaded through tonic"
        height, width = SYNTHETIC_SHAPE
        return (
            f"synthetic {self._dataset} {label}events on a "
            f"{height}x{width} sensor (offline; no real recording)"
        )

    @property
    def dataset(self) -> str:
        """Return the dataset registry key."""
        return self._dataset

    @property
    def split(self) -> str:
        """Return the dataset split this source serves."""
        return self._split

    @property
    def origin(self) -> str:
        """Return the backend name (``tonic`` or ``synthetic``)."""
        return self._origin
