"""Node-kind dispatch table and scheduling helpers for the interpreter."""

from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch

from snn_interpreter.nir_bridge import ops_linear, ops_neuron

#: An operation: ``(node, summed_input, state) -> (output, state, membrane)``.
NodeOp = Callable[
    [Any, torch.Tensor, Any],
    Tuple[torch.Tensor, Any, Any],
]

INPUT = "Input"
OUTPUT = "Output"
DELAY = "Delay"

#: Every parametric node kind the reference interpreter implements.
OPS: Dict[str, NodeOp] = {
    "Affine": ops_linear.apply_affine,
    "Linear": ops_linear.apply_linear,
    "Conv2d": ops_linear.apply_conv2d,
    "Conv1d": ops_linear.apply_conv1d,
    "Flatten": ops_linear.apply_flatten,
    "AvgPool2d": ops_linear.apply_avgpool,
    "SumPool2d": ops_linear.apply_sumpool,
    "LI": ops_neuron.apply_li,
    "LIF": ops_neuron.apply_lif,
    "IF": ops_neuron.apply_if,
    "CubaLIF": ops_neuron.apply_cuba,
    "Threshold": ops_neuron.apply_threshold,
    "Scale": ops_neuron.apply_scale,
    "Output": ops_neuron.apply_identity,
}

#: Node kinds whose output is a binary spike train.
SPIKING_KINDS = ("Threshold", "LIF", "IF", "CubaLIF")
#: Node kinds that expose a membrane potential trace.
INTEGRATOR_KINDS = ("LI", "LIF", "IF", "CubaLIF")
#: Node kinds that consume multi-dimensional (spatial) frames.
SPATIAL_KINDS = ("Conv2d", "AvgPool2d", "SumPool2d", "Flatten")
#: Every node kind the interpreter accepts, including the virtual nodes.
SUPPORTED_KINDS = frozenset(OPS) | {INPUT, DELAY}


def reduced_edges(
    edges: Sequence[Tuple[str, str]], is_delay: Callable[[str], bool]
) -> List[Tuple[str, str]]:
    """Return ``edges`` without the inputs of Delay nodes.

    A Delay node emits the previous step's value, so its own input only
    refreshes the buffer after the step and must not order it.
    """
    return [
        (source, target)
        for source, target in edges
        if not is_delay(target)
    ]


def children_map(
    names: Sequence[str], edges: Sequence[Tuple[str, str]]
) -> Dict[str, List[str]]:
    """Return a child adjacency map keyed by node name."""
    children: Dict[str, List[str]] = {name: [] for name in names}
    for source, target in edges:
        children[source].append(target)
    return children


def drain(
    ready: List[str],
    children: Dict[str, List[str]],
    pending: Dict[str, int],
) -> List[str]:
    """Kahn-drain ``ready`` into a deterministic execution order."""
    order: List[str] = []
    while ready:
        name = ready.pop(0)
        order.append(name)
        for child in children[name]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    return order


def sum_inputs(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return the elementwise sum of the incoming edge values."""
    if not values:
        raise ValueError("node received no inputs")
    total = values[0]
    for value in values[1:]:
        total = total + value
    return total


def stack(
    frames: Dict[str, List[torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """Stack per-step frame lists into ``[T, ...]`` trace tensors."""
    return {name: torch.stack(items) for name, items in frames.items()}
