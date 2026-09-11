"""``spikeforge-io``: dependency-light I/O adapters and the windowing surface.

PT-W7's second deliverable. The distribution is deliberately thin: it adapts a
recorded numeric source (CSV, JSON, NPY, in-memory, or any object implementing
the dataset hook) into the core frozen window/normalize/encode contract, and
replays it into ``spikeforge-serve``'s ``/v1/stream``.

The windowing *implementation* stays in core
(:mod:`spikeforge.streaming.window_spec`) because core must never depend on a
satellite; :mod:`spikeforge_io.windowing` re-exports that one canonical
contract. No pandas, no fastapi, no client SDK is imported here.

    from spikeforge_io import CsvAdapter, replay_to_stream

    adapter = CsvAdapter("recording.csv")
    frames = adapter.stream()          # [N, D]
    windows = adapter.windows(spec)    # [W, L, D], z-scored by the frozen spec
"""

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
from spikeforge_io.windowing import (
    WINDOW_SPEC_VERSION,
    WindowSpec,
    delta_over_window,
    encode_windows,
    fit_window_spec,
    normalize_windows,
    window_spec_from_bundle,
    window_stream,
    windows,
)

__all__ = [
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
