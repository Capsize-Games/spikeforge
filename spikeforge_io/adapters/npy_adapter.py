"""Adapter reading a ``.npy`` array into a stream via numpy.

Split out of the ``adapters`` package root so each adapter lives in its
own file. numpy is imported lazily; its absence is a typed
:class:`~spikeforge_io.errors.AdapterDependencyError`.
"""

import torch

from spikeforge_io.adapters.base import StreamAdapter, as_stream
from spikeforge_io.errors import AdapterDependencyError, AdapterFormatError


class NpyAdapter(StreamAdapter):
    """Read a ``.npy`` array into an ``[N, D]`` stream via numpy."""

    def __init__(self, path: str) -> None:
        """Record the ``path`` to the ``.npy`` file."""
        self.source = path

    def stream(self) -> torch.Tensor:
        """Return the array as a ``[N, D]`` tensor, or report numpy absent."""
        try:
            import numpy as np
        except ImportError:
            raise AdapterDependencyError("numpy", "a .npy stream") from None
        try:
            payload = np.load(self.source, allow_pickle=False)
        except (OSError, ValueError) as error:
            raise AdapterFormatError(
                self.source, f"cannot read .npy: {error}"
            ) from None
        return as_stream(payload, self.source)
