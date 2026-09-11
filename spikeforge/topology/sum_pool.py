"""A 2D sum-pooling module (PyTorch only ships average/max pooling)."""

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as functional

_Size = Union[int, Tuple[int, int]]


def _as_pair(value: _Size) -> Tuple[int, int]:
    """Return ``value`` widened to an ``(h, w)`` integer pair."""
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


class SumPool2d(nn.Module):
    """Sum each 2D pooling window, including padding as zeros."""

    def __init__(
        self,
        kernel_size: _Size,
        stride: Union[int, None] = None,
        padding: Union[int, Tuple[int, int]] = 0,
    ) -> None:
        """Store the pooling geometry and the window area."""
        super().__init__()
        self._kernel = kernel_size
        self._stride = stride
        self._padding = padding
        height, width = _as_pair(kernel_size)
        self._area = height * width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the summed pooling windows of ``x``."""
        pooled = functional.avg_pool2d(
            x,
            self._kernel,
            self._stride,
            self._padding,
            count_include_pad=True,
        )
        return pooled * self._area
