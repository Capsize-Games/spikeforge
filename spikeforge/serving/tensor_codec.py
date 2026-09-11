"""JSON-safe encode/decode of the tensors an inference session carries.

Carried state is a nested structure of tensors, tuples, dicts, and ``None``.
``json`` cannot represent any of that, so :func:`encode` walks the structure,
stores each tensor with its exact dtype and shape (through the NIR
:mod:`~spikeforge.nir_bridge.array_codec`), and tags each tuple; :func:`decode`
rebuilds the identical structure on request. Round-tripping is exact: no float
approximation and no list-instead-of-tuple drift.
"""

from typing import Any, Optional

import numpy as np
import torch

from spikeforge.nir_bridge import array_codec

#: Key marking a mapping as an encoded tuple.
TUPLE_KEY = "__tuple__"


def _to_array(tensor: torch.Tensor) -> np.ndarray:
    """Return ``tensor`` as a numpy array, falling back to float32.

    A few dtypes (for example ``bfloat16``) have no numpy equivalent; those
    are widened so the state is still serializable rather than silently lost.
    """
    value = tensor.detach().cpu()
    try:
        return value.numpy()
    except TypeError:
        return value.float().numpy()


def encode(value: Any) -> Any:
    """Return ``value`` with tensors and tuples encoded for JSON."""
    if isinstance(value, torch.Tensor):
        return array_codec.encode(_to_array(value))
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return {TUPLE_KEY: [encode(item) for item in value]}
    if isinstance(value, list):
        return [encode(item) for item in value]
    return value


def decode(value: Any, device: Optional[torch.device] = None) -> Any:
    """Return ``value`` with tagged tensors and tuples restored.

    ``device`` places every rebuilt tensor on that device; ``None`` keeps the
    default CPU placement.
    """
    if isinstance(value, dict):
        if value.get(array_codec.ARRAY_KEY):
            tensor = torch.as_tensor(array_codec.decode(value))
            return tensor if device is None else tensor.to(device)
        if TUPLE_KEY in value:
            return tuple(decode(item, device) for item in value[TUPLE_KEY])
        return {key: decode(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item, device) for item in value]
    return value


def move(value: Any, device: torch.device) -> Any:
    """Return ``value`` with every tensor moved to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move(item, device) for item in value)
    if isinstance(value, list):
        return [move(item, device) for item in value]
    return value
