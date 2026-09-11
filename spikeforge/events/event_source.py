"""Event-modality samples: real ``tonic`` datasets or explicit synthetic.

The loader itself lives in :mod:`spikeforge.data.event_loader`; this
source decides which backend serves a sample. It prefers the real dataset
whenever ``tonic`` can load it, and falls back to the deterministic
generators in :mod:`spikeforge.events.synthetic` only for the offline
path. The chosen backend is recorded in :attr:`origin` and
:attr:`description`, so a synthetic stream is never presented as a real
recording.
"""

from typing import Any, Dict, Optional, Tuple

import torch

from spikeforge.data import event_loader
from spikeforge.data.datasets import dataset_available, dataset_info
from spikeforge.events import synthetic
from spikeforge.events.event_sample import EventSample

#: Backend names reported by :attr:`EventSampleSource.origin`.
TONIC = "tonic"
SYNTHETIC = "synthetic"
#: Square sensor every synthetic stream is generated on, so one sample can
#: drive both feature (28*28) and spatial (28x28) input stages.
SYNTHETIC_SHAPE: Tuple[int, int] = (28, 28)
#: Distinct synthetic samples the offline source serves before repeating.
SYNTHETIC_SIZE = 64
#: Time bins synthetic streams span.
SYNTHETIC_STEPS = 10


class EventSampleSource:
    """Serve ``(EventSample, label)`` pairs for a dataset key and index.

    ``synthetic_only`` forces the offline generator even when ``tonic`` is
    installed, which keeps tests and demos network-free. Otherwise the real
    dataset is used whenever its loader is available.
    """

    def __init__(
        self,
        dataset: str = "n_mnist",
        synthetic_only: bool = False,
        num_steps: int = SYNTHETIC_STEPS,
        save_to: Optional[str] = None,
    ) -> None:
        """Choose the backend and cache the dataset's class count."""
        self._dataset = dataset
        self._num_steps = int(num_steps)
        self._save_to = save_to
        self._classes = dataset_info(dataset)[0]
        self._origin = self._pick(synthetic_only)

    def _pick(self, synthetic_only: bool) -> str:
        """Return the backend name this source will use."""
        if synthetic_only or not dataset_available(self._dataset):
            return SYNTHETIC
        return TONIC

    def clamp(self, index: int) -> int:
        """Wrap an index into the backend's valid range."""
        if self._origin == SYNTHETIC:
            return int(index) % SYNTHETIC_SIZE
        return max(0, int(index))

    def load(self, index: int) -> Tuple[EventSample, int]:
        """Return the ``(sample, label)`` pair for ``index``."""
        if self._origin == TONIC:
            return event_loader.load_event_pair(
                self._dataset, index, self._num_steps, self._save_to
            )
        return self._synthetic(index), self._synthetic_label(index)

    def _synthetic(self, index: int) -> EventSample:
        """Return a deterministic two-polarity stream for ``index``."""
        on = synthetic.moving_dot(
            num_steps=self._num_steps, shape=SYNTHETIC_SHAPE,
            start=(2.0 + index % 8, 2.0 + index % 6),
            velocity=(0.8, 1.0), seed=index,
        )
        off = synthetic.moving_dot(
            num_steps=self._num_steps, shape=SYNTHETIC_SHAPE,
            start=(20.0 - index % 6, 20.0 - index % 8),
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
            "description": self.description,
        }

    @property
    def description(self) -> str:
        """Return a one-line, human-readable provenance description."""
        if self._origin == TONIC:
            return f"{self._dataset} events loaded through tonic"
        height, width = SYNTHETIC_SHAPE
        return (
            f"synthetic {self._dataset} events on a {height}x{width} sensor "
            "(offline; no real recording)"
        )

    @property
    def dataset(self) -> str:
        """Return the dataset registry key."""
        return self._dataset

    @property
    def origin(self) -> str:
        """Return the backend name (``tonic`` or ``synthetic``)."""
        return self._origin
