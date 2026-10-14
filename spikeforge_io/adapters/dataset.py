"""Adapter hook for third-party dataset objects.

Split out of the ``adapters`` package root so each adapter lives in its
own file. The hook is deliberately duck-typed so a future dataset lib can
join without this package depending on it.
"""

from typing import TYPE_CHECKING, Any, Optional, Tuple

from spikeforge_io.adapters.base import (
    StreamAdapter,
    as_stream,
    as_stream_numpy,
)
from spikeforge_io.errors import AdapterFormatError

if TYPE_CHECKING:
    import numpy as np
    import torch

#: Methods the dataset-adapter hook recognises, in probing order.
DATASET_METHODS: Tuple[str, ...] = ("to_tensor", "array", "numpy")


class DatasetAdapter(StreamAdapter):
    """Adapt any object implementing the documented dataset hook.

    The hook is intentionally tiny and library-agnostic: an object is adapt-
    able when it exposes ``to_tensor()`` (or ``array()``/``numpy()``) returning
    an array-like, **or** when it supports the sequence protocol
    (``__len__`` + ``__getitem__``). This is the seam a future dataset
    distribution plugs into without ``spikeforge-io`` importing it.
    """

    def __init__(self, dataset: Any, source: Optional[str] = None) -> None:
        """Wrap ``dataset`` and label it with ``source`` or its type name."""
        self._dataset = dataset
        self.source = source or type(dataset).__name__

    def _payload(self) -> Any:
        """Return the array-like the dataset exposes, or a typed refusal."""
        for name in DATASET_METHODS:
            method = getattr(self._dataset, name, None)
            if callable(method):
                return method()
        if hasattr(self._dataset, "__len__") and hasattr(
            self._dataset, "__getitem__"
        ):
            indices = range(len(self._dataset))
            return [self._dataset[index] for index in indices]
        raise AdapterFormatError(
            self.source,
            "dataset must expose to_tensor()/array() or the sequence protocol",
        )

    def stream(self) -> "torch.Tensor":
        """Return the dataset's samples as a ``[N, D]`` tensor."""
        return as_stream(self._payload(), self.source)

    def stream_numpy(self) -> "np.ndarray":
        """Return the dataset's samples as a ``[N, D]`` array."""
        return as_stream_numpy(self._payload(), self.source)
