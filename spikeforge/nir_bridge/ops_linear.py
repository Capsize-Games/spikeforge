"""Affine, convolution, flatten and pooling node operations."""

from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as functional

from spikeforge.nir_bridge.ops_common import as_pair, as_tensor

Result = Tuple[torch.Tensor, Any, Any]


def apply_affine(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return ``x @ weight.T + bias``."""
    weight = as_tensor(node.weight, x)
    bias = as_tensor(node.bias, x)
    return functional.linear(x, weight, bias), None, None


def apply_linear(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return ``x @ weight.T`` for a bias-free linear node."""
    weight = as_tensor(node.weight, x)
    return functional.linear(x, weight, None), None, None


def _optional_bias(node: Any, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the node bias as a tensor, or ``None`` when it has none."""
    if node.bias is None:
        return None
    return as_tensor(node.bias, x)


def apply_conv2d(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return the 2D convolution of ``x`` with the node's weight and bias."""
    convolved = functional.conv2d(
        x,
        as_tensor(node.weight, x),
        _optional_bias(node, x),
        stride=as_pair(node.stride),
        padding=as_pair(node.padding),
        dilation=as_pair(node.dilation),
        groups=int(node.groups),
    )
    return convolved, None, None


def apply_conv1d(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return the 1D convolution of ``x`` with the node's weight and bias.

    ``nir.Conv1d`` uses channels-first ``[B, C, L]`` frames, matching
    ``nn.Conv1d``; its stride/padding/dilation are scalars (or a padding
    string), so they pass straight through to the functional call.
    """
    convolved = functional.conv1d(
        x,
        as_tensor(node.weight, x),
        _optional_bias(node, x),
        stride=node.stride,
        padding=node.padding,
        dilation=node.dilation,
        groups=int(node.groups),
    )
    return convolved, None, None


def apply_flatten(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return ``x`` flattened from ``start_dim`` through ``end_dim``."""
    flattened = torch.flatten(x, int(node.start_dim), int(node.end_dim))
    return flattened, None, None


def apply_avgpool(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return the average-pooled ``x`` with padded cells counted."""
    pooled = functional.avg_pool2d(
        x,
        as_pair(node.kernel_size),
        as_pair(node.stride),
        as_pair(node.padding),
        count_include_pad=True,
    )
    return pooled, None, None


def apply_sumpool(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return summed pools: average windows times the window area."""
    kernel = as_pair(node.kernel_size)
    pooled = functional.avg_pool2d(
        x,
        kernel,
        as_pair(node.stride),
        as_pair(node.padding),
        count_include_pad=True,
    )
    return pooled * float(kernel[0] * kernel[1]), None, None
