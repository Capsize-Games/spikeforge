"""Adapter reading a ``.npy`` array into a stream via numpy.

Split out of the ``adapters`` package root so each adapter lives in its
own file. numpy is imported lazily; its absence is a typed
:class:`~spikeforge_io.errors.AdapterDependencyError`.
"""

from typing import TYPE_CHECKING

from spikeforge_io.adapters.base import (
    StreamAdapter,
    _require_numpy,
    as_stream,
    as_stream_numpy,
)
from spikeforge_io.errors import AdapterFormatError

if TYPE_CHECKING:
    import numpy as np
    import torch


class NpyAdapter(StreamAdapter):
    """Read a ``.npy`` array into an ``[N, D]`` stream via numpy."""

    def __init__(self, path: str) -> None:
        """Record the ``path`` to the ``.npy`` file."""
        self.source = path

    def _array(self) -> "np.ndarray":
        """Return the file's array, or report numpy absent/unreadable."""
        numpy = _require_numpy("a .npy stream")
        try:
            return numpy.load(self.source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AdapterFormatError(
                self.source, f"cannot read .npy: {error}"
            ) from None

    def stream(self) -> "torch.Tensor":
        """Return the array as a ``[N, D]`` tensor, or report numpy absent."""
        return as_stream(self._array(), self.source)

    def stream_numpy(self) -> "np.ndarray":
        """Return the array as a ``[N, D]`` array, or report numpy absent."""
        return as_stream_numpy(self._array(), self.source)
