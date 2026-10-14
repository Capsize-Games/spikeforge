"""Drive ``libnir_c_runtime.so`` in-process through ctypes.

Why ctypes and not a file/subprocess handoff: a serving session is
stateful — LI membranes and Delay buffers carry across steps inside
the C runtime — and the C API offers no state serialization, so a
handoff would need a newly written C driver binary plus per-step IPC
to shuttle opaque state back and forth. ctypes keeps the runtime's
buffers where they belong (in C, across calls) with the standard
library alone, so the core distribution gains no new dependency.

One instance owns one graph and one runtime and is not thread-safe,
which matches how sessions use it: ``ServingService`` already guards
each session behind its own lock.
"""

import ctypes
import os
from typing import Any, List, Optional, Tuple

from spikeforge.serving.c_abi import CGraph, CRuntime, CTensor
from spikeforge.serving.c_runtime_bind import bind
from spikeforge.serving.c_runtime_env import (
    BUILD_HINT,
    LIB_ENV_VAR,
    libc_free,
    resolve_lib_path,
)
from spikeforge.serving.errors import (
    CRuntimeExecutionError,
    CRuntimeLibraryError,
)


def _check_frame_shape(shape: Tuple[int, ...]) -> None:
    """Raise unless ``shape`` fits a C tensor (1 to 3 data dims)."""
    if 1 <= len(shape) <= 3 and all(side > 0 for side in shape):
        return
    raise ValueError(
        f"frame shape {tuple(shape)} is not 1 to 3 dense dims"
    )


def _input_tensor(data: int, shape: Tuple[int, ...]) -> CTensor:
    """Return an unowned tensor viewing ``data`` as ``shape``."""
    size = 1
    for side in shape:
        size *= side
    frame = CTensor()
    frame.data = ctypes.cast(data, ctypes.POINTER(ctypes.c_float))
    frame.ndim = len(shape)
    for index, side in enumerate(shape):
        frame.shape[index] = side
    frame.size = size
    frame.capacity = size
    frame.owns_data = 0
    return frame


class NirCLibrary:
    """One loaded C graph and its stateful runtime.

    ``load_graph`` parses a ``graph_summary``-shaped JSON document and
    initializes the runtime; :meth:`run_frame` steps one frame with the
    runtime's carried state intact; :meth:`reset` re-initializes that
    state; :meth:`close` releases everything. The parsed JSON stays
    alive until :meth:`close` so no loaded pointer can dangle.
    """

    def __init__(self, path: Optional[str] = None) -> None:
        """Load the shared library from ``path`` or the environment."""
        resolved = resolve_lib_path(path)
        if not resolved:
            raise CRuntimeLibraryError(
                None,
                "no library path was given and "
                f"{LIB_ENV_VAR} is unset; {BUILD_HINT}",
            )
        self._lib = self._open(resolved)
        self._free = libc_free()
        self._err = ctypes.c_void_p(None)
        self._json: Any = None
        self._graph: Optional[CGraph] = None
        self._runtime: Optional[CRuntime] = None

    def __enter__(self) -> "NirCLibrary":
        """Return this library for a ``with`` block."""
        return self

    def __exit__(self, *exc: Any) -> None:
        """Release the graph and runtime on block exit."""
        self.close()

    @property
    def loaded(self) -> bool:
        """Return True once a graph and runtime are live."""
        return self._graph is not None and self._runtime is not None

    def load_graph(self, graph_json: str) -> None:
        """Parse ``graph_json`` and initialize its runtime.

        Any previously loaded graph is released first, so calling
        this twice reloads rather than leaks. A failed load releases
        whatever it built and leaves nothing live.
        """
        self.close()
        parsed = self._parse(graph_json)
        graph = self._load(parsed)
        try:
            self._runtime = self._init(graph)
        except CRuntimeExecutionError:
            self._lib.nir_graph_free(ctypes.byref(graph))
            self._lib.json_free(parsed)
            raise
        self._json = parsed
        self._graph = graph

    def run_frame(
        self, data: int, shape: Tuple[int, ...]
    ) -> Tuple[List[float], Tuple[int, ...]]:
        """Step one frame and return ``(values, shape)`` of the readout.

        ``data`` is the address of ``prod(shape)`` contiguous float32
        samples the caller keeps alive for the call; stepping carries
        the runtime's LI and Delay state into the next call.
        """
        self._require_loaded("run")
        _check_frame_shape(shape)
        assert self._runtime is not None  # narrowed by _require_loaded
        frame = _input_tensor(data, shape)
        readout = CTensor()
        self._run_once(frame, readout)
        values = [readout.data[i] for i in range(readout.size)]
        out_shape = tuple(readout.shape[i] for i in range(readout.ndim))
        self._lib.nir_tensor_free(ctypes.byref(readout))
        return values, out_shape

    def reset(self) -> None:
        """Re-initialize the runtime state, keeping the loaded graph."""
        if not self.loaded:
            return
        assert self._graph is not None and self._runtime is not None
        self._lib.nir_runtime_free(ctypes.byref(self._runtime))
        try:
            self._runtime = self._init(self._graph)
        except CRuntimeExecutionError:  # pragma: no cover - defensive
            self._runtime = None
            raise

    def close(self) -> None:
        """Release the runtime, graph, and parsed JSON, if any."""
        if self._runtime is not None:
            self._lib.nir_runtime_free(ctypes.byref(self._runtime))
            self._runtime = None
        if self._graph is not None:
            self._lib.nir_graph_free(ctypes.byref(self._graph))
            self._graph = None
        if self._json is not None:
            self._lib.json_free(self._json)
            self._json = None

    @staticmethod
    def _open(resolved: str) -> ctypes.CDLL:
        """Return the bound shared library at ``resolved``."""
        if not os.path.isfile(resolved):
            raise CRuntimeLibraryError(
                resolved, f"no such file; {BUILD_HINT}"
            )
        try:
            library = ctypes.CDLL(resolved)
        except OSError as error:
            raise CRuntimeLibraryError(
                resolved, f"the loader refused it: {error}"
            ) from None
        bind(library)
        return library

    def _err_ref(self) -> Any:
        """Return the error out-param, or NULL without a libc free."""
        if self._free is None:
            return None
        self._err.value = None
        return ctypes.byref(self._err)

    def _failure(self, operation: str) -> CRuntimeExecutionError:
        """Return the typed error for a failed ``operation`` call.

        The C message is copied, then released through libc's free;
        without a message the failure is still typed, only terser.
        """
        address: Any = self._err.value
        self._err.value = None
        if address is None:
            return CRuntimeExecutionError(operation, "unknown error")
        message = ctypes.string_at(address).decode()
        if self._free is not None:
            self._free(ctypes.c_void_p(address))
        return CRuntimeExecutionError(operation, message)

    def _require_loaded(self, operation: str) -> None:
        """Raise unless a graph and runtime are live."""
        if not self.loaded:
            raise CRuntimeExecutionError(
                operation, "no graph is loaded"
            )

    def _run_once(self, frame: CTensor, readout: CTensor) -> None:
        """Step ``frame`` once, filling ``readout`` or raising typed."""
        assert self._runtime is not None  # callers require loaded first
        ok = self._lib.nir_runtime_run(
            ctypes.byref(self._runtime), ctypes.byref(frame), 1,
            ctypes.byref(readout), self._err_ref(),
        )
        if not ok:
            self._lib.nir_tensor_free(ctypes.byref(readout))
            raise self._failure("run")

    def _parse(self, graph_json: str) -> Any:
        """Return the parsed ``graph_json``, refusing invalid JSON."""
        parsed = self._lib.json_parse(
            graph_json.encode("utf-8"), self._err_ref()
        )
        if not parsed:
            raise self._failure("parse")
        return parsed

    def _load(self, parsed: Any) -> CGraph:
        """Return the graph loaded from one parsed JSON value."""
        graph = CGraph()
        if not self._lib.nir_graph_load(
            parsed, ctypes.byref(graph), self._err_ref()
        ):
            self._lib.nir_graph_free(ctypes.byref(graph))
            self._lib.json_free(parsed)
            raise self._failure("load")
        return graph

    def _init(self, graph: CGraph) -> CRuntime:
        """Return the runtime initialized for ``graph``.

        A failed init frees the runtime itself (see ``runtime.c``),
        so this side only reports it; the graph stays the caller's.
        """
        runtime = CRuntime()
        if not self._lib.nir_runtime_init(
            ctypes.byref(runtime), ctypes.byref(graph), self._err_ref()
        ):
            raise self._failure("init")
        return runtime
