"""Dependency-light adapters for ingesting numeric streams into the contract.

Every adapter answers one question — "what is this recorded source, as an
``[N, D]`` numeric stream?" — and nothing more. Windowing, normalization, and
encoding are then the shared core contract (see
:mod:`spikeforge_io.windowing`), so an adapter can never invent its own
preprocessing. Only the standard library and torch are used for CSV/JSON and
in-memory data; numpy is imported lazily and only for ``.npy``, and its absence
is a typed :class:`~spikeforge_io.errors.AdapterDependencyError` rather than an
``ImportError`` at the call site.

The dataset-adapter **hook** (:class:`DatasetAdapter`) is deliberately
duck-typed: any object exposing ``to_tensor()``/``array()`` or the sequence
protocol (``__len__`` + ``__getitem__``) adapts, so a future dataset lib can
join without this module depending on it.
"""

import csv
import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from spikeforge_io.errors import (
    AdapterDependencyError,
    AdapterFormatError,
    AdapterShapeError,
    AdapterUnsupportedError,
)

#: Methods the dataset-adapter hook recognises, in probing order.
DATASET_METHODS: Tuple[str, ...] = ("to_tensor", "array", "numpy")

#: Keys a JSON object may use to carry the stream payload.
JSON_KEYS: Tuple[str, ...] = ("data", "stream", "values")


def as_stream(payload: Any, source: str) -> torch.Tensor:
    """Return ``payload`` as an ``[N, D]`` float32 tensor, or a typed error.

    A 1-D payload is promoted to a single channel; a 2-D payload is taken as
    ``[N, D]``; anything else is refused with
    :class:`~spikeforge_io.errors.AdapterShapeError`. A non-numeric payload is
    a :class:`~spikeforge_io.errors.AdapterFormatError`.
    """
    try:
        tensor = torch.as_tensor(payload, dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise AdapterFormatError(
            source, f"stream is not numeric: {error}"
        ) from None
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


class InMemoryAdapter(StreamAdapter):
    """Adapt an already-materialized payload (tensor, nested list, ndarray)."""

    def __init__(self, payload: Any, source: str = "<memory>") -> None:
        """Wrap ``payload`` and label it with ``source``."""
        self._payload = payload
        self.source = source

    def stream(self) -> torch.Tensor:
        """Return the payload as a ``[N, D]`` tensor."""
        return as_stream(self._payload, self.source)


def _read_rows(path: str, delimiter: str) -> List[List[str]]:
    """Read a CSV file into raw rows, or raise a typed format error."""
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            return list(csv.reader(handle, delimiter=delimiter))
    except OSError as error:
        raise AdapterFormatError(path, f"cannot read: {error}") from None


def _parse_csv_row(path: str, index: int, row: Sequence[str]) -> List[float]:
    """Parse one CSV row into floats, naming the first bad field."""
    values: List[float] = []
    for column, cell in enumerate(row):
        text = cell.strip()
        if not text:
            continue
        try:
            values.append(float(text))
        except ValueError:
            raise AdapterFormatError(
                path,
                f"row {index} column {column} is not numeric: {cell!r}",
                position=index,
            ) from None
    if not values:
        raise AdapterFormatError(
            path, f"row {index} has no numeric fields", position=index
        )
    return values


class CsvAdapter(StreamAdapter):
    """Read a ``.csv`` file of numeric rows into an ``[N, D]`` stream."""

    def __init__(
        self,
        path: str,
        *,
        delimiter: str = ",",
        has_header: bool = True,
        channels: Optional[int] = None,
    ) -> None:
        """Record the ``path``, delimiter, header flag, and expected width."""
        self.source = path
        self._delimiter = delimiter
        self._has_header = has_header
        self._channels = channels

    def stream(self) -> torch.Tensor:
        """Return the CSV's numeric rows as a ``[N, D]`` tensor."""
        rows = _read_rows(self.source, self._delimiter)
        if self._has_header and rows:
            rows = rows[1:]
        if not rows:
            raise AdapterShapeError(self.source, "CSV has no data rows")
        parsed = [
            _parse_csv_row(self.source, index, row)
            for index, row in enumerate(rows)
        ]
        width = len(parsed[0])
        for index, row in enumerate(parsed):
            if len(row) != width:
                raise AdapterFormatError(
                    self.source,
                    f"row {index} has {len(row)} fields, expected {width}",
                    position=index,
                )
        if self._channels is not None and width != self._channels:
            raise AdapterShapeError(
                self.source,
                f"expected {self._channels} channels, got {width}",
            )
        return as_stream(parsed, self.source)


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

    def stream(self) -> torch.Tensor:
        """Return the JSON stream as a ``[N, D]`` tensor."""
        payload = self._load()
        if self._key is not None:
            if not isinstance(payload, dict) or self._key not in payload:
                raise AdapterFormatError(
                    self.source, f"JSON object has no {self._key!r} key"
                )
            payload = payload[self._key]
        else:
            payload = _json_payload(payload, self.source)
        try:
            return as_stream(payload, self.source)
        except AdapterFormatError as error:
            raise AdapterFormatError(
                self.source, f"JSON rows are not rectangular: {error.detail}"
            ) from None


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

    def stream(self) -> torch.Tensor:
        """Return the dataset's samples as a ``[N, D]`` tensor."""
        return as_stream(self._payload(), self.source)


#: Adapter factories keyed by a lower-cased filename suffix.
FACTORIES: Dict[str, Callable[..., StreamAdapter]] = {
    ".csv": CsvAdapter,
    ".tsv": lambda path, **kwargs: CsvAdapter(path, delimiter="\t", **kwargs),
    ".json": JsonAdapter,
    ".npy": NpyAdapter,
}


def adapter_for(path: str, **kwargs: Any) -> StreamAdapter:
    """Return the adapter registered for ``path``'s suffix.

    An unrecognised suffix is refused with
    :class:`~spikeforge_io.errors.AdapterUnsupportedError` naming the supported
    formats, so a caller never gets a silent misparse.
    """
    lowered = path.lower()
    for suffix, factory in FACTORIES.items():
        if lowered.endswith(suffix):
            return factory(path, **kwargs)
    raise AdapterUnsupportedError(
        path,
        "no adapter for this suffix; supported: "
        + ", ".join(sorted(FACTORIES)),
    )
