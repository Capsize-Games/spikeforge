"""DoS bounds for client-controlled topology sizes.

``TrainConfig`` bounds its own scalars, but ``topology_params`` is a
free-form mapping that flows into :func:`build_topology` and from
there into layer allocation. This module is the last check before any
tensor exists: every size-bearing integer must be positive and within
its cap, so ``hidden=10**9`` fails validation instead of OOMing the
box.
"""

from typing import Any, Dict, Mapping, Tuple

#: Largest hidden-layer width any topology may allocate.
MAX_HIDDEN = 16384
#: Largest class count any topology may allocate.
MAX_NUM_CLASSES = 10000
#: Largest flat feature count (``fc_*``/``recurrent_net`` input_size).
MAX_INPUT_FEATURES = 1024 * 1024
#: Largest sensor side (conv ``input_size`` int or ``(H, W)`` dim).
MAX_SENSOR_SIDE = 1024
#: Largest input channel count (covers RGB plus multispectral).
MAX_IN_CHANNELS = 1024
#: Largest conv channel count in either direction.
MAX_CHANNELS = 4096
#: Longest sequence any sequence topology may unroll.
MAX_SEQ_LENGTH = 4096
#: Widest per-token feature width.
MAX_FEATURES = 16384
#: Largest attention vocabulary.
MAX_VOCAB = 262144
#: Widest attention embedding.
MAX_EMBED_DIM = 4096
#: Most attention heads.
MAX_NUM_HEADS = 64

#: (min, max) for each size-bearing int, keyed by parameter name.
_BOUNDS: Dict[str, Tuple[int, int]] = {
    "hidden": (1, MAX_HIDDEN),
    "num_classes": (1, MAX_NUM_CLASSES),
    "in_channels": (1, MAX_IN_CHANNELS),
    "channels": (1, MAX_CHANNELS),
    "seq_length": (1, MAX_SEQ_LENGTH),
    "features": (1, MAX_FEATURES),
    "vocab": (1, MAX_VOCAB),
    "embed_dim": (1, MAX_EMBED_DIM),
    "num_heads": (1, MAX_NUM_HEADS),
}


def _is_int(value: Any) -> bool:
    """Return True for a genuine int, excluding booleans."""
    return isinstance(value, int) and not isinstance(value, bool)


def _describe(value: Any) -> str:
    """Return a bounded rendering of a rejected value for errors."""
    if _is_int(value):
        return repr(value)
    return type(value).__name__


def _check_int(name: str, value: Any, low: int, high: int) -> None:
    """Raise when ``value`` is not an int inside ``[low, high]``."""
    if not _is_int(value) or not low <= value <= high:
        raise ValueError(
            f"topology param {name!r} must be an int in "
            f"[{low}, {high}], got {_describe(value)}"
        )


def _check_conv_size(value: Any) -> None:
    """Raise unless ``value`` is a sensor side/pair within bounds."""
    if _is_int(value):
        sides = (value, value)
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        sides = (value[0], value[1])
    else:
        raise ValueError(
            "conv_net input_size must be an int side or an (H, W) "
            f"pair, got {_describe(value)}"
        )
    for side in sides:
        _check_int("input_size", side, 1, MAX_SENSOR_SIDE)


def validate_params(name: str, params: Mapping[str, Any]) -> None:
    """Raise when a size-bearing param of ``name`` is absurd.

    Runs after defaults merge and before anything is built, so a
    rejection always precedes allocation.
    """
    for key, (low, high) in _BOUNDS.items():
        if key in params:
            _check_int(key, params[key], low, high)
    if "input_size" not in params:
        return
    if name == "conv_net":
        _check_conv_size(params["input_size"])
    else:
        _check_int(
            "input_size", params["input_size"], 1, MAX_INPUT_FEATURES
        )
