"""ctypes layouts mirroring the ``nir-c-runtime`` C headers.

Each struct here mirrors one ``nir-c-runtime``
``include/nir_c_runtime/*.h`` declaration field for field, so the
mapping stays reviewable against the headers it tracks (see
``TRACKED_COMMIT`` in :mod:`spikeforge_targets.c_runtime`). Only the
layouts Python must allocate cross this boundary: the graph and runtime
handles are otherwise opaque, and JSON values never leave C as structs.
"""

from spikeforge.serving.c_abi.graph import CGraph
from spikeforge.serving.c_abi.runtime import CRuntime
from spikeforge.serving.c_abi.tensor import CTensor

__all__ = [
    "CGraph",
    "CRuntime",
    "CTensor",
]
