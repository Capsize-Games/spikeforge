"""Adapter for an already-materialized in-memory payload.

Split out of the ``adapters`` package root so each adapter lives in its
own file.
"""

from typing import TYPE_CHECKING, Any

from spikeforge_io.adapters.base import (
    StreamAdapter,
    as_stream,
    as_stream_numpy,
)

if TYPE_CHECKING:
    import numpy as np
    import torch


class InMemoryAdapter(StreamAdapter):
    """Adapt an already-materialized payload (tensor, nested list, ndarray)."""

    def __init__(self, payload: Any, source: str = "<memory>") -> None:
        """Wrap ``payload`` and label it with ``source``."""
        self._payload = payload
        self.source = source

    def stream(self) -> "torch.Tensor":
        """Return the payload as a ``[N, D]`` tensor."""
        return as_stream(self._payload, self.source)

    def stream_numpy(self) -> "np.ndarray":
        """Return the payload as a ``[N, D]`` array without torch."""
        return as_stream_numpy(self._payload, self.source)
