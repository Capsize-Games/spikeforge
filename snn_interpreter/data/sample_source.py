"""Dataset-backed source of single samples for viewing and encoding."""

from typing import Tuple

import torch

from snn_interpreter.data.datasets import build_dataset


class SampleSource:
    """Expose single transformed images and labels from a registry set."""

    def __init__(
        self,
        dataset: str = "mnist",
        train: bool = True,
        download: bool = True,
    ) -> None:
        """Load the registry dataset backing this source."""
        self._dataset = dataset
        self._data = build_dataset(dataset, train=train)

    def clamp(self, index: int) -> int:
        """Wrap an index into the valid range [0, len)."""
        size = len(self._data)
        return int(index) % size if size else 0

    def image(self, index: int) -> torch.Tensor:
        """Return the transformed image tensor [1,28,28] in [0,1]."""
        image, _ = self._data[self.clamp(index)]
        return image

    def label(self, index: int) -> int:
        """Return the integer label for a sample index."""
        _, label = self._data[self.clamp(index)]
        return int(label)

    def __len__(self) -> int:
        """Return the number of samples in the source dataset."""
        return len(self._data)

    @property
    def size(self) -> Tuple[int, int]:
        """Return the (H, W) size of every sample."""
        return (28, 28)

    @property
    def dataset(self) -> str:
        """Return the dataset registry key."""
        return self._dataset
