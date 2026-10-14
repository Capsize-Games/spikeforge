"""Windowed, z-scored samples with a class label and an anomaly flag."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class StreamDataset:
    """Windowed, z-scored samples with a class label and an anomaly flag."""

    windows: torch.Tensor
    labels: torch.Tensor
    anomaly: torch.Tensor

    def __len__(self) -> int:
        """Return the number of windows."""
        return int(self.windows.size(0))
