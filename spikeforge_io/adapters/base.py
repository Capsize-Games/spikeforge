"""Base class and payload coercion for recorded-stream adapters.

Split out of the ``adapters`` package root so each adapter lives in its
own file. Subclasses implement :meth:`StreamAdapter.stream`; windowing and
encoding always flow through the one core contract.

Neither torch nor numpy is imported at module level: both load lazily, so
ingestion works with either absent, and a missing framework is a typed
:class:`~spikeforge_io.errors.AdapterDependencyError`.
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from spikeforge_io.errors import (
    AdapterDependencyError,
    AdapterFormatError,
    AdapterShapeError,
)

if TYPE_CHECKING:
    import numpy as np
    import torch


def _require_torch(feature: str) -> Any:
    """Return the torch module, or report it absent with a typed error."""
    try:
        import torch
    except ImportError:
        raise AdapterDependencyError(
            "torch", feature, hint="install torch to use the torch view"
        ) from None
    return torch


def _require_numpy(feature: str) -> Any:
    """Return the numpy module, or report it absent with a typed error."""
    try:
        import numpy
    except ImportError:
        raise AdapterDependencyError("numpy", feature) from None
    return numpy


def _coerce(payload: Any, source: str) -> "torch.Tensor":
    """Return ``payload`` as a float32 tensor, or raise a format error."""
    torch = _require_torch("a torch stream")
    try:
        return torch.as_tensor(payload, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise AdapterFormatError(
            source, f"stream is not numeric: {error}"
        ) from None


def _coerce_numpy(payload: Any, source: str) -> "np.ndarray":
    """Return ``payload`` as a float32 array, or raise a format error."""
    numpy = _require_numpy("a numpy stream")
    if payload is None:
        raise AdapterFormatError(source, "stream is not numeric: None")
    try:
        return numpy.asarray(payload, dtype=numpy.float32)
    except (TypeError, ValueError) as error:
        raise AdapterFormatError(
            source, f"stream is not numeric: {error}"
        ) from None


def as_stream(payload: Any, source: str) -> "torch.Tensor":
    """Return ``payload`` as an ``[N, D]`` float32 tensor, or a typed error.

    A 1-D payload is promoted to a single channel; a 2-D payload is taken as
    ``[N, D]``; anything else is refused with
    :class:`~spikeforge_io.errors.AdapterShapeError`. A non-numeric payload is
    a :class:`~spikeforge_io.errors.AdapterFormatError`.
    """
    tensor = _coerce(payload, source)
    if tensor.dim() == 0:
        return tensor.reshape(1, 1)
    if tensor.dim() == 1:
        return tensor.unsqueeze(-1)
    if tensor.dim() != 2:
        raise AdapterShapeError(
            source, f"a stream must be 1-D or 2-D, got {tuple(tensor.shape)}"
        )
    return tensor


def as_stream_numpy(payload: Any, source: str) -> "np.ndarray":
    """Return ``payload`` as an ``[N, D]`` float32 array, or a typed error.

    The numpy twin of :func:`as_stream`: identical shaping and errors, but
    no torch import, so ingestion works with torch absent.
    """
    array = _coerce_numpy(payload, source)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim != 2:
        raise AdapterShapeError(
            source, f"a stream must be 1-D or 2-D, got {tuple(array.shape)}"
        )
    return array


class StreamAdapter(ABC):
    """Base class for every recorded-stream adapter.

    Subclasses implement :meth:`stream`; :meth:`stream_numpy` defaults to
    converting that tensor, and the built-in adapters override it with a
    torch-free path. :meth:`windows` and :meth:`encoded` are provided here
    so windowing/encoding always flow through the one core contract.
    """

    #: Human-readable source label used in error messages.
    source: str = "<adapter>"

    @abstractmethod
    def stream(self) -> "torch.Tensor":
        """Return the source as a ``[N, D]`` float32 tensor."""

    def stream_numpy(self) -> "np.ndarray":
        """Return the source as a ``[N, D]`` float32 array.

        The default converts :meth:`stream`, so it needs torch installed;
        every built-in adapter overrides this with a torch-free path.
        """
        return as_stream_numpy(self.stream(), self.source)

    def windows(self, spec: Any) -> "torch.Tensor":
        """Return the z-scored ``[W, L, D]`` windows of :meth:`stream`."""
        spec.validate()
        return spec.windows(self.stream())

    def encoded(
        self, spec: Any, encode_spec: Any, **kwargs: Any
    ) -> "torch.Tensor":
        """Return windows encoded with the shared window->spike contract."""
        from spikeforge_io.windowing import encode_windows

        return encode_windows(self.windows(spec), encode_spec, **kwargs)

    def __len__(self) -> int:
        """Return the number of samples in the stream."""
        return int(self.stream_numpy().shape[0])
