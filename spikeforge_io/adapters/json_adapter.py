"""Adapter reading a ``.json`` file into a stream.

Split out of the ``adapters`` package root so each adapter lives in its
own file.
"""

import json
from typing import TYPE_CHECKING, Any, Callable, Optional, Tuple

from spikeforge_io.adapters.base import (
    StreamAdapter,
    as_stream,
    as_stream_numpy,
)
from spikeforge_io.errors import AdapterFormatError

if TYPE_CHECKING:
    import numpy as np
    import torch

#: Keys a JSON object may use to carry the stream payload.
JSON_KEYS: Tuple[str, ...] = ("data", "stream", "values")


def _json_payload(payload: Any, path: str) -> Any:
    """Return the stream payload inside a JSON document to adapt."""
    if isinstance(payload, dict):
        for key in (JSON_KEYS + ("payload",)):
            if key in payload:
                return payload[key]
        raise AdapterFormatError(
            path, f"JSON object has no stream key ({', '.join(JSON_KEYS)})"
        )
    return payload


class JsonAdapter(StreamAdapter):
    """Read a ``.json`` file (rows, scalars, or an object) as a stream."""

    def __init__(self, path: str, *, key: Optional[str] = None) -> None:
        """Record the ``path`` and an optional explicit object ``key``."""
        self.source = path
        self._key = key

    def _load(self) -> Any:
        """Return the raw JSON payload, or raise a typed format error."""
        try:
            with open(self.source, encoding="utf-8") as handle:
                return json.load(handle)
        except OSError as error:
            raise AdapterFormatError(
                self.source, f"cannot read: {error}"
            ) from None
        except json.JSONDecodeError as error:
            raise AdapterFormatError(
                self.source, f"malformed JSON: {error}"
            ) from None

    def _payload(self) -> Any:
        """Return the selected stream payload, or a typed error."""
        payload = self._load()
        if self._key is not None:
            if not isinstance(payload, dict) or self._key not in payload:
                raise AdapterFormatError(
                    self.source, f"JSON object has no {self._key!r} key"
                )
            return payload[self._key]
        return _json_payload(payload, self.source)

    def _adapt(self, coerce: Callable[[Any, str], Any]) -> Any:
        """Coerce the payload, blaming ragged rows on the JSON shape."""
        payload = self._payload()
        try:
            return coerce(payload, self.source)
        except AdapterFormatError as error:
            raise AdapterFormatError(
                self.source, f"JSON rows are not rectangular: {error.detail}"
            ) from None

    def stream(self) -> "torch.Tensor":
        """Return the JSON stream as a ``[N, D]`` tensor."""
        return self._adapt(as_stream)

    def stream_numpy(self) -> "np.ndarray":
        """Return the JSON stream as a ``[N, D]`` array."""
        return self._adapt(as_stream_numpy)
