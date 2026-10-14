"""The ``nir_tensor_t`` layout from ``tensor.h``."""

import ctypes


class CTensor(ctypes.Structure):
    """A dense float tensor, up to 3 data dimensions.

    Mirrors ``nir_tensor_t`` exactly: the data pointer, the dimension
    count, the fixed 3-slot shape, the element count, the writable
    capacity, and the ownership flag ``nir_tensor_free`` honors.
    """

    _fields_ = [
        ("data", ctypes.POINTER(ctypes.c_float)),
        ("ndim", ctypes.c_size_t),
        ("shape", ctypes.c_size_t * 3),
        ("size", ctypes.c_size_t),
        ("capacity", ctypes.c_size_t),
        ("owns_data", ctypes.c_int),
    ]
