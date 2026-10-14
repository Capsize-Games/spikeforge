"""The ``nir_graph_t`` layout from ``graph.h``."""

import ctypes


class CGraph(ctypes.Structure):
    """A loaded NIR graph: nodes, edges, and I/O indices.

    Mirrors ``nir_graph_t`` exactly. Python only allocates this struct
    for ``nir_graph_load`` to fill and passes it on opaquely; the node
    and edge buffers stay owned by C and are freed by
    ``nir_graph_free``.
    """

    _fields_ = [
        ("nodes", ctypes.c_void_p),
        ("node_count", ctypes.c_size_t),
        ("edges", ctypes.c_void_p),
        ("edge_count", ctypes.c_size_t),
        ("input_index", ctypes.c_size_t),
        ("output_index", ctypes.c_size_t),
    ]
