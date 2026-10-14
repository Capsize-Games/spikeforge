"""Declare the C signatures the ctypes binding calls.

Each ``_bind_*`` function covers one ``nir-c-runtime`` header's calls;
:func:`bind` applies all three to a freshly loaded shared library.
"""

import ctypes

from spikeforge.serving.c_abi import CGraph, CRuntime, CTensor

#: Error out-param type. The C headers declare ``char **``, but this
#: side only holds the address (to copy, then free, the message), so a
#: ``void *`` holder is ABI-identical and unambiguous to release.
_ErrorRef = ctypes.POINTER(ctypes.c_void_p)


def bind(library: ctypes.CDLL) -> None:
    """Declare every C signature this binding calls on ``library``."""
    _bind_json(library)
    _bind_graph(library)
    _bind_runtime(library)


def _bind_json(library: ctypes.CDLL) -> None:
    """Declare the ``json.h`` parse and free calls."""
    library.json_parse.argtypes = [ctypes.c_char_p, _ErrorRef]
    library.json_parse.restype = ctypes.c_void_p
    library.json_free.argtypes = [ctypes.c_void_p]
    library.json_free.restype = None


def _bind_graph(library: ctypes.CDLL) -> None:
    """Declare the ``graph.h`` load and free calls."""
    library.nir_graph_load.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(CGraph), _ErrorRef
    ]
    library.nir_graph_load.restype = ctypes.c_int
    library.nir_graph_free.argtypes = [ctypes.POINTER(CGraph)]
    library.nir_graph_free.restype = None


def _bind_runtime(library: ctypes.CDLL) -> None:
    """Declare the ``runtime.h``/``tensor.h`` step and free calls."""
    library.nir_runtime_init.argtypes = [
        ctypes.POINTER(CRuntime), ctypes.POINTER(CGraph), _ErrorRef
    ]
    library.nir_runtime_init.restype = ctypes.c_int
    library.nir_runtime_free.argtypes = [ctypes.POINTER(CRuntime)]
    library.nir_runtime_free.restype = None
    library.nir_runtime_run.argtypes = [
        ctypes.POINTER(CRuntime), ctypes.POINTER(CTensor),
        ctypes.c_size_t, ctypes.POINTER(CTensor), _ErrorRef,
    ]
    library.nir_runtime_run.restype = ctypes.c_int
    library.nir_tensor_free.argtypes = [ctypes.POINTER(CTensor)]
    library.nir_tensor_free.restype = None
