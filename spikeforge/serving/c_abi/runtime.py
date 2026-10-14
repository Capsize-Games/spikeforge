"""The ``nir_runtime_t`` layout from ``runtime.h``."""

import ctypes


class CRuntime(ctypes.Structure):
    """One graph's execution order and temporal state.

    Mirrors ``nir_runtime_t`` exactly: the graph pointer, the
    topological order, and the LI-membrane and Delay buffers with
    their primed flags. Python only allocates this struct for
    ``nir_runtime_init`` to fill; stepping keeps every buffer in C,
    which is what lets state carry across steps without copying.
    """

    _fields_ = [
        ("graph", ctypes.c_void_p),
        ("order", ctypes.c_void_p),
        ("li_state", ctypes.c_void_p),
        ("li_primed", ctypes.c_void_p),
        ("delay_buffer", ctypes.c_void_p),
        ("delay_primed", ctypes.c_void_p),
    ]
