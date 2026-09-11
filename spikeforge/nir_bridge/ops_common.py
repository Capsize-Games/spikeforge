"""Tensor helpers shared by the reference interpreter's node operations."""

from typing import Any, Tuple

import numpy as np
import torch


def as_tensor(value: Any, reference: torch.Tensor) -> torch.Tensor:
    """Return ``value`` as a tensor on ``reference``'s dtype and device."""
    array = np.asarray(value)
    return torch.as_tensor(
        array, dtype=reference.dtype, device=reference.device
    )


def as_pair(value: Any) -> Tuple[int, int]:
    """Return a scalar or length-2 sequence widened to an int pair."""
    items = np.asarray(value).reshape(-1)
    if items.size == 1:
        return int(items[0]), int(items[0])
    return int(items[0]), int(items[1])
