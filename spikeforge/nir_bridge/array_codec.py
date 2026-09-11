"""Encode numpy values inside an NIR payload as JSON-safe mappings.

A node's ``to_dict`` output contains numpy arrays (weights, thresholds,
scalar parameters) that ``json`` cannot represent directly. This codec stores
each array as a tagged mapping carrying its dtype and shape, so reloading
reconstructs the exact array rather than an approximated list of floats.
Everything else is passed through unchanged.
"""

from typing import Any, Dict

import numpy as np

#: Key marking a mapping as an encoded numpy array.
ARRAY_KEY = "__ndarray__"


def _encode_array(array: np.ndarray) -> Dict[str, Any]:
    """Return the tagged, JSON-able mapping describing ``array``."""
    return {
        ARRAY_KEY: True,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "data": array.ravel().tolist(),
    }


def encode(value: Any) -> Any:
    """Return ``value`` with numpy arrays and scalars encoded for JSON."""
    if isinstance(value, np.ndarray):
        return _encode_array(value)
    if isinstance(value, np.generic):
        return _encode_array(np.asarray(value))
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    return value


def _is_array(mapping: Dict[str, Any]) -> bool:
    """Return True when ``mapping`` is a tagged array record."""
    return bool(mapping.get(ARRAY_KEY))


def _decode_array(mapping: Dict[str, Any]) -> np.ndarray:
    """Return the numpy array described by a tagged ``mapping``."""
    dtype = np.dtype(mapping["dtype"])
    return np.asarray(mapping["data"], dtype=dtype).reshape(mapping["shape"])


def decode(value: Any) -> Any:
    """Return ``value`` with every tagged array restored to numpy."""
    if isinstance(value, dict):
        if _is_array(value):
            return _decode_array(value)
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
