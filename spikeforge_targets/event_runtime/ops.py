"""Event-driven linear and convolution operators over sparse spike frames.

Each operator multiplies only the nonzero entries of its input, so it pays for
spikes rather than zeros while reproducing the dense result exactly (a sparse
matmul over the nonzero entries is the same arithmetic). It returns the output
together with the number of synaptic operations it actually performed, which is
the honest op count the accounting layer consumes.
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as functional

from spikeforge_targets.event_runtime.errors import UnsupportedKindError
from spikeforge_targets.event_runtime.spike_view import SparseSpikes

#: Stage kinds with a genuine event-driven implementation.
SPARSE_KINDS: Tuple[str, ...] = ("linear", "conv2d")


def apply(
    kind: str, module: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Dispatch ``x`` through the event-driven operator for ``kind``."""
    if kind == "linear":
        return linear_ops(module, x)
    return conv2d_ops(module, x)


def connected_inputs(kind: str, module: nn.Module) -> int:
    """Return the input fan-in of one output element of ``module``."""
    if kind == "linear":
        return int(module.in_features)
    height, width = _pair(module.kernel_size)
    return int(module.in_channels // module.groups) * height * width


def _pair(value: object) -> Tuple[int, int]:
    """Return ``value`` widened to an integer ``(h, w)`` pair."""
    if isinstance(value, int):
        return value, value
    if isinstance(value, (tuple, list)):
        return int(value[0]), int(value[1])
    raise UnsupportedKindError(f"non-integer geometry {value!r}")


def linear_ops(
    module: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Return ``(output, sop)`` of a linear stage over sparse ``x``."""
    flat = x.reshape(-1, x.shape[-1])
    spikes = SparseSpikes.from_dense(flat)
    if spikes.nnz == 0:
        out = torch.zeros(
            flat.shape[0], int(module.out_features),
            dtype=flat.dtype, device=flat.device,
        )
    else:
        out = torch.sparse.mm(spikes.torch_sparse(), module.weight.t())
    if module.bias is not None:
        out = out + module.bias
    shaped = out.reshape(*x.shape[:-1], int(module.out_features))
    return shaped, spikes.nnz * int(module.out_features)


def conv2d_ops(
    module: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Return ``(output, sop)`` of a conv2d stage over sparse ``x``."""
    if int(module.groups) != 1:
        raise UnsupportedKindError(f"conv2d(groups={int(module.groups)})")
    kernel = _pair(module.kernel_size)
    patches = functional.unfold(
        x, module.kernel_size, dilation=module.dilation,
        padding=module.padding, stride=module.stride,
    )
    weight = module.weight.reshape(int(module.out_channels), -1)
    outs, sop = _sparse_batches(weight, patches)
    height, width = _output_hw(x, module, kernel)
    shaped = outs.reshape(
        outs.shape[0], int(module.out_channels), height, width
    )
    if module.bias is not None:
        shaped = shaped + module.bias.reshape(1, -1, 1, 1)
    return shaped, sop


def _sparse_batches(
    weight: torch.Tensor, patches: torch.Tensor
) -> Tuple[torch.Tensor, int]:
    """Return the stacked per-batch sparse products and the synaptic tally."""
    outs = []
    sop = 0
    for index in range(int(patches.shape[0])):
        sparse = patches[index].to_sparse()
        outs.append(_product(weight, sparse))
        sop += int(sparse.values().numel()) * int(weight.shape[0])
    return torch.stack(outs), sop


def _product(weight: torch.Tensor, sparse: torch.Tensor) -> torch.Tensor:
    """Return ``weight @ sparse`` densified, zero when nothing is active."""
    if int(sparse.values().numel()) == 0:
        return torch.zeros(
            weight.shape[0], sparse.shape[1],
            dtype=weight.dtype, device=weight.device,
        )
    return torch.sparse.mm(weight, sparse)


def _output_hw(
    x: torch.Tensor, module: nn.Module, kernel: Tuple[int, int]
) -> Tuple[int, int]:
    """Return the spatial output size of ``module`` applied to ``x``."""
    height = _axis(
        x.shape[-2], kernel[0], module.padding, module.dilation,
        module.stride, 0,
    )
    width = _axis(
        x.shape[-1], kernel[1], module.padding, module.dilation,
        module.stride, 1,
    )
    return height, width


def _axis(
    size: int, kernel: int, padding: object, dilation: object,
    stride: object, index: int,
) -> int:
    """Return one spatial output extent from a conv geometry."""
    pad = _pair(padding)[index]
    dil = _pair(dilation)[index]
    step = _pair(stride)[index]
    return (int(size) + 2 * pad - dil * (kernel - 1) - 1) // step + 1
