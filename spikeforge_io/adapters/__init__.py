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
join without this package depending on it.

This package is the import root: the base class lives in
:mod:`spikeforge_io.adapters.base`, each adapter in its own module, and the
suffix registry below. Everything is re-exported here so importers keep
using this path.
"""

from typing import Any, Callable, Dict

from spikeforge_io.adapters.base import StreamAdapter, as_stream
from spikeforge_io.adapters.csv_adapter import CsvAdapter
from spikeforge_io.adapters.dataset import DATASET_METHODS, DatasetAdapter
from spikeforge_io.adapters.json_adapter import JSON_KEYS, JsonAdapter
from spikeforge_io.adapters.memory import InMemoryAdapter
from spikeforge_io.adapters.npy_adapter import NpyAdapter
from spikeforge_io.errors import AdapterUnsupportedError

__all__ = [
    "DATASET_METHODS",
    "FACTORIES",
    "JSON_KEYS",
    "CsvAdapter",
    "DatasetAdapter",
    "InMemoryAdapter",
    "JsonAdapter",
    "NpyAdapter",
    "StreamAdapter",
    "adapter_for",
    "as_stream",
]

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
