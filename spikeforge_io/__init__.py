"""``spikeforge-io``: dependency-light I/O adapters and the windowing surface.

PT-W7's second deliverable. The distribution is deliberately thin: it adapts a
recorded numeric source (CSV, JSON, NPY, in-memory, or any object implementing
the dataset hook) into the core frozen window/normalize/encode contract, and
replays it into ``spikeforge-serve``'s ``/v1/stream``.

The windowing *implementation* stays in core
(:mod:`spikeforge.streaming.window_spec`) because core must never depend on a
satellite; :mod:`spikeforge_io.windowing` re-exports that one canonical
contract. No pandas, no fastapi, no client SDK is imported here. The windowing
names below resolve lazily (PEP 562) because they are the one surface that
needs torch installed; everything else imports torch-free.

    from spikeforge_io import CsvAdapter, replay_to_stream

    adapter = CsvAdapter("recording.csv")
    frames = adapter.stream()          # [N, D]
    windows = adapter.windows(spec)    # [W, L, D], z-scored by the frozen spec
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _distribution_version
from typing import Any, FrozenSet

from spikeforge_io.adapters import (
    DATASET_METHODS,
    FACTORIES,
    CsvAdapter,
    DatasetAdapter,
    InMemoryAdapter,
    JsonAdapter,
    NpyAdapter,
    StreamAdapter,
    adapter_for,
    as_stream,
    as_stream_numpy,
)
from spikeforge_io.errors import (
    AdapterDependencyError,
    AdapterError,
    AdapterFormatError,
    AdapterShapeError,
    AdapterUnsupportedError,
    IoError,
)
from spikeforge_io.replay import (
    ReplayReport,
    replay,
    replay_to_stream,
    stream_messages,
)

#: The torch-backed windowing names, resolved lazily (see :func:`__getattr__`
#: below) so adapter ingestion works without torch installed.
_WINDOWING_EXPORTS: FrozenSet[str] = frozenset(
    {
        "WINDOW_SPEC_VERSION",
        "WindowSpec",
        "delta_over_window",
        "encode_windows",
        "fit_window_spec",
        "normalize_windows",
        "window_spec_from_bundle",
        "window_stream",
        "windows",
    }
)


def __getattr__(name: str) -> Any:
    """Return a lazily imported windowing re-export (PEP 562)."""
    if name in _WINDOWING_EXPORTS:
        from spikeforge_io import windowing

        return getattr(windowing, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


try:
    #: Read from installed metadata so it cannot drift from the wheel.
    __version__ = _distribution_version("spikeforge-io")
except PackageNotFoundError:  # pragma: no cover - uninstalled source tree
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "DATASET_METHODS",
    "FACTORIES",
    "WINDOW_SPEC_VERSION",
    "AdapterDependencyError",
    "AdapterError",
    "AdapterFormatError",
    "AdapterShapeError",
    "AdapterUnsupportedError",
    "CsvAdapter",
    "DatasetAdapter",
    "InMemoryAdapter",
    "IoError",
    "JsonAdapter",
    "NpyAdapter",
    "ReplayReport",
    "StreamAdapter",
    "WindowSpec",
    "adapter_for",
    "as_stream",
    "as_stream_numpy",
    "delta_over_window",
    "encode_windows",
    "fit_window_spec",
    "normalize_windows",
    "replay",
    "replay_to_stream",
    "stream_messages",
    "window_spec_from_bundle",
    "window_stream",
    "windows",
]
