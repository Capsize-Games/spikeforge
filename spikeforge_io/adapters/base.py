"""Base class and payload coercion for recorded-stream adapters.

Split out of the ``adapters`` package root so each adapter lives in its
own file. Subclasses implement :meth:`StreamAdapter.stream`; windowing and
encoding always flow through the one core contract.
"""

from abc import ABC, abstractmethod
from typing import Any

import torch

from spikeforge_io.errors import AdapterFormatError, AdapterShapeError


def _coerce(payload: Any, source: str) -> torch.Tensor:
    """Return ``payload`` as a float32 tensor, or raise a format error."""
    try:
        return torch.as_tensor(payload, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise AdapterFormatError(
            source, f"stream is not numeric: {error}"
        ) from None


def as_stream(payload: Any, source: str) -> torch.Tensor:
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


class StreamAdapter(ABC):
    """Base class for every recorded-stream adapter.

    Subclasses implement :meth:`stream`; :meth:`windows` and :meth:`encoded`
    are provided here so windowing/encoding always flow through the one core
    contract.
    """

    #: Human-readable source label used in error messages.
    source: str = "<adapter>"

    @abstractmethod
    def stream(self) -> torch.Tensor:
        """Return the source as a ``[N, D]`` float32 tensor."""

    def windows(self, spec: Any) -> torch.Tensor:
        """Return the z-scored ``[W, L, D]`` windows of :meth:`stream`."""
        spec.validate()
        return spec.windows(self.stream())

    def encoded(
        self, spec: Any, encode_spec: Any, **kwargs: Any
    ) -> torch.Tensor:
        """Return windows encoded with the shared window->spike contract."""
        from spikeforge_io.windowing import encode_windows

        return encode_windows(self.windows(spec), encode_spec, **kwargs)

    def __len__(self) -> int:
        """Return the number of samples in the stream."""
        return int(self.stream().size(0))
