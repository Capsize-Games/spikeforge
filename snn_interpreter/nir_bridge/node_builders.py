"""Build NIR nodes for the parameterised (non-neuron) stage kinds.

A linear stage maps to ``nir.Affine`` when it carries a bias and to
``nir.Linear`` when it does not, matching each node's semantics. When the
built module is available its tensors and convolution settings are copied
directly; otherwise a structurally coherent graph is emitted with
zero-initialised placeholder weights.
"""

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from snn_interpreter.nir_bridge.mapped_node import MappedNode
from snn_interpreter.nir_bridge.require import require_node


def _np(tensor: Any) -> np.ndarray:
    """Detach a torch tensor into a numpy array."""
    return tensor.detach().cpu().numpy()


def _opt_np(tensor: Optional[Any]) -> Optional[np.ndarray]:
    """Convert ``tensor`` to numpy, preserving ``None``."""
    return None if tensor is None else _np(tensor)


def _pair(value: Any) -> Optional[Tuple[int, int]]:
    """Normalise a scalar or length-2 sequence into an int pair."""
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    items = tuple(int(item) for item in value)
    if len(items) == 1:
        return (items[0], items[0])
    return (items[0], items[1])


def _placeholder_linear(params: Mapping[str, Any]) -> Tuple[Any, Any]:
    """Return zero-initialised ``(weight, bias)`` for a structural export."""
    out_features = int(params["out_features"])
    in_features = int(params["in_features"])
    weight = np.zeros((out_features, in_features), np.float32)
    bias = None
    if bool(params.get("bias", True)):
        bias = np.zeros(out_features, np.float32)
    return weight, bias


def linear_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``Affine``/``Linear`` node rendering a linear stage."""
    if submodule is None:
        weight, bias = _placeholder_linear(params)
    else:
        weight = _np(submodule.weight)
        bias = _opt_np(submodule.bias)
    if bias is None:
        return MappedNode(name, require_node("Linear", "linear")(weight))
    return MappedNode(name, require_node("Affine", "linear")(weight, bias))


def _conv_weight_shape(
    params: Mapping[str, Any], groups: int
) -> Tuple[int, int, int, int]:
    """Return the ``(out, in, kh, kw)`` weight shape for a conv stage."""
    kernel = _pair(params["kernel_size"])
    return (
        int(params["out_channels"]),
        int(params["in_channels"]) // groups,
        kernel[0],
        kernel[1],
    )


def _module_conv_settings(submodule: Any) -> Tuple[Any, ...]:
    """Return conv arguments copied from a built torch module."""
    return (
        _np(submodule.weight),
        submodule.stride,
        submodule.padding,
        submodule.dilation,
        int(submodule.groups),
        _opt_np(submodule.bias),
    )


def _structural_conv_settings(params: Mapping[str, Any]) -> Tuple[Any, ...]:
    """Return conv arguments with zero placeholder weights."""
    groups = int(params.get("groups", 1))
    weight = np.zeros(_conv_weight_shape(params, groups), np.float32)
    bias = np.zeros(int(params["out_channels"]), np.float32)
    return (
        weight,
        params.get("stride", 1),
        params.get("padding", 0),
        params.get("dilation", 1),
        groups,
        bias,
    )


def conv_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``Conv2d`` node rendering a convolution stage."""
    cls = require_node("Conv2d", "conv2d")
    if submodule is None:
        settings = _structural_conv_settings(params)
    else:
        settings = _module_conv_settings(submodule)
    return MappedNode(name, cls(None, *settings))


def flatten_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``Flatten`` node rendering a flatten stage."""
    cls = require_node("Flatten", "flatten")
    start = int(params.get("start_dim", 1))
    end = int(params.get("end_dim", -1))
    return MappedNode(name, cls({"input": None}, start, end))


def _pool_node(
    name: str, params: Mapping[str, Any], primitive: str, kind: str
) -> MappedNode:
    """Return a pooling node with kernel/stride/padding pairs."""
    cls = require_node(primitive, kind)
    kernel = _pair(params["kernel_size"])
    stride = _pair(params.get("stride")) or kernel
    padding = _pair(params.get("padding", 0)) or (0, 0)
    node = cls(np.array(kernel), np.array(stride), np.array(padding))
    return MappedNode(name, node)


def avgpool_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``AvgPool2d`` node rendering an average-pool stage."""
    return _pool_node(name, params, "AvgPool2d", "avgpool2d")


def sumpool_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``SumPool2d`` node rendering a sum-pool stage."""
    return _pool_node(name, params, "SumPool2d", "sumpool2d")
