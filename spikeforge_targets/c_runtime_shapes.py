"""Node-level conformance of an export summary against the C loader.

Each check mirrors one assumption in ``nir-c-runtime`` (tracked commit in
:mod:`spikeforge_targets.c_runtime`): the ``KIND_TABLE`` in ``graph.h``,
the weight/pair parsing in ``src/graph.c``, the dense-only convolution
and all-dims flatten in ``src/ops.c``, and the hardcoded one-step delay
in ``src/runtime.c``. A ``None`` reason means the node conforms.
"""

import math
from typing import Any, Mapping, Optional, Tuple

from spikeforge_targets.c_runtime import KNOWN_GAPS, SUPPORTED
from spikeforge_targets.c_runtime_violation import CRuntimeViolation

#: An ``[h, w]`` integer pair from a summary's parameters.
Pair = Tuple[int, int]


def _is_int(value: Any) -> bool:
    """Return True for a genuine int (bools excluded)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Return True for a genuine int or float (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any) -> Optional[float]:
    """Return ``value`` as a float, or None when it is not a number."""
    if not _is_number(value):
        return None
    return float(value)


def _pair(value: Any) -> Optional[Pair]:
    """Return ``value`` as an int pair, or None when it is not one."""
    if not isinstance(value, list) or len(value) != 2:
        return None
    if not _is_int(value[0]) or not _is_int(value[1]):
        return None
    return (int(value[0]), int(value[1]))


def _positive_pair(value: Any) -> Optional[Pair]:
    """Return ``value`` as a positive int pair, or None if it is not."""
    pair = _pair(value)
    if pair is None or pair[0] <= 0 or pair[1] <= 0:
        return None
    return pair


def _envelope_data(value: Any) -> Optional[Any]:
    """Return the payload of a valid quantized envelope, else None.

    Mirrors ``weight_view()``: ``dtype`` is ``int8``/``uint8``, ``scale``
    is positive and finite, ``zero_point`` is finite, and ``data`` is a
    JSON array the shape check then descends into.
    """
    if not isinstance(value, dict):
        return None
    data = value.get("data")
    scale = _as_float(value.get("scale"))
    zero = _as_float(value.get("zero_point"))
    if not isinstance(data, list) or value.get("dtype") not in (
        "int8",
        "uint8",
    ):
        return None
    if scale is None or not math.isfinite(scale):
        return None
    if scale <= 0.0 or zero is None:
        return None
    if not math.isfinite(zero):
        return None
    return data


def _raw_shape(value: Any, depth: int) -> Optional[Tuple[int, ...]]:
    """Return the rectangular numeric dims of ``value`` at ``depth``."""
    if depth == 0:
        return () if _is_number(value) else None
    if not isinstance(value, list) or not value:
        return None
    first = _raw_shape(value[0], depth - 1)
    if first is None:
        return None
    for item in value[1:]:
        if _raw_shape(item, depth - 1) != first:
            return None
    return (len(value), *first)


def _array_shape(value: Any, depth: int) -> Optional[Tuple[int, ...]]:
    """Return the dims of a raw or quantized weight array at ``depth``."""
    if isinstance(value, dict):
        payload = _envelope_data(value)
        return None if payload is None else _raw_shape(payload, depth)
    return _raw_shape(value, depth)


def _affine_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why an Affine node's params break the C loader, if any."""
    shape = _array_shape(params.get("weight"), 2)
    if shape is None:
        return "Affine weight must be a 2D [out, in] float array"
    bias = _array_shape(params.get("bias"), 1)
    if bias is None or bias[0] != shape[0]:
        return "Affine bias must be a 1D array matching out_features"
    return None


def _conv_weight_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why a Conv2d weight or bias breaks the C loader, if any."""
    shape = _array_shape(params.get("weight"), 4)
    if shape is None:
        return "Conv2d weight must be a 4D [out, in, kh, kw] float array"
    bias = params.get("bias")
    if bias is None:
        return None
    bias_shape = _array_shape(bias, 1)
    if bias_shape is None or bias_shape[0] != shape[0]:
        return "Conv2d bias must be null or a 1D array of out_channels"
    return None


def _conv_grid_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why a Conv2d grid breaks the C convolution, if any."""
    if _positive_pair(params.get("stride")) is None:
        return "Conv2d stride must be a positive [h, w] int pair"
    padding = _pair(params.get("padding"))
    if padding is None or min(padding) < 0:
        return "Conv2d padding must be a non-negative [h, w] int pair"
    if _positive_pair(params.get("dilation")) is None:
        return "Conv2d dilation must be a positive [h, w] int pair"
    groups = params.get("groups")
    if not _is_int(groups) or groups != 1:
        return "Conv2d groups must be 1; the C convolution is dense-only"
    return None


def _pool_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why a pooling node's params break the C loader, if any."""
    if _positive_pair(params.get("kernel_size")) is None:
        return "pool kernel_size must be a positive [h, w] int pair"
    if _positive_pair(params.get("stride")) is None:
        return "pool stride must be a positive [h, w] int pair"
    padding = _pair(params.get("padding"))
    if padding is None or min(padding) < 0:
        return "pool padding must be a non-negative [h, w] int pair"
    return None


def _flatten_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why a Flatten node's dims diverge from the C op, if any."""
    start = params.get("start_dim")
    end = params.get("end_dim")
    if not _is_int(start) or not _is_int(end):
        return "Flatten start_dim and end_dim must be ints"
    if start != 1 or end != -1:
        return (
            "Flatten must cover every non-batch dim "
            "(start_dim=1, end_dim=-1); the C op flattens all dims"
        )
    return None


def _delay_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why a Delay node's length diverges from the C op, if any."""
    steps = _as_float(params.get("delay"))
    if steps is None or steps != 1.0:
        return "Delay must be exactly one step; the C runtime hardcodes 1"
    return None


def _li_reason(params: Mapping[str, Any]) -> Optional[str]:
    """Return why an LI node's params break the C op, if any."""
    tau = _as_float(params.get("tau"))
    if tau is None or not math.isfinite(tau):
        return "LI tau must be a finite number"
    if tau <= 0.0:
        return "LI tau must be positive; the C op rejects tau <= 0"
    for key in ("r", "v_leak"):
        if not _is_number(params.get(key)):
            return f"LI {key} must be a number"
    return None


def _scalar_reason(
    kind: str, params: Mapping[str, Any], key: str
) -> Optional[str]:
    """Return why a Threshold/Scale node's params break the C op, if any."""
    if not _is_number(params.get(key)):
        return f"{kind} {key} must be a number"
    return None


def _node_reason(kind: str, params: Mapping[str, Any]) -> Optional[str]:
    """Return why a supported kind's params break the C loader, if any."""
    if kind == "Affine":
        return _affine_reason(params)
    if kind == "Conv2d":
        return _conv_weight_reason(params) or _conv_grid_reason(params)
    if kind in ("AvgPool2d", "SumPool2d"):
        return _pool_reason(params)
    if kind == "Flatten":
        return _flatten_reason(params)
    if kind == "Delay":
        return _delay_reason(params)
    if kind == "LI":
        return _li_reason(params)
    if kind == "Threshold":
        return _scalar_reason(kind, params, "threshold")
    if kind == "Scale":
        return _scalar_reason(kind, params, "scale")
    return None


def _kind_reason(kind: str) -> Optional[str]:
    """Return why a node kind fails C loading, or None when supported."""
    if kind in SUPPORTED:
        return None
    hint = KNOWN_GAPS.get(kind)
    if hint is not None:
        return f"unsupported node kind {kind!r}: {hint}"
    return f"unsupported node kind {kind!r}: the C loader has no such node"


def check_node(
    name: str, kind: str, params: Mapping[str, Any]
) -> Optional[CRuntimeViolation]:
    """Return the violation for one summary node, or None when conforming.

    An unsupported kind is reported on its own; parameter checks run
    only for kinds the C loader implements.
    """
    reason = _kind_reason(kind)
    if reason is None:
        reason = _node_reason(kind, params)
    if reason is None:
        return None
    return CRuntimeViolation(name=name, kind=kind, reason=reason)
