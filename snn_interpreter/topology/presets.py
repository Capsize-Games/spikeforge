"""Ready-made topology specs shared by tests, training, and validation."""

from typing import Any, Dict, List

from snn_interpreter.topology.edge import Edge
from snn_interpreter.topology.spec import TopologySpec, chain
from snn_interpreter.topology.stage import Stage


def _linear_params(in_features: int, out_features: int) -> Dict[str, Any]:
    return {
        "in_features": in_features,
        "out_features": out_features,
    }


def _conv_params(
    in_channels: int, out_channels: int, kernel_size: int = 3
) -> Dict[str, Any]:
    return {
        "in_channels": in_channels,
        "out_channels": out_channels,
        "kernel_size": kernel_size,
        "padding": 1,
    }


def fc_legacy(
    hidden: int = 128, beta: float = 0.5,
    num_classes: int = 10, input_size: int = 28 * 28,
) -> TopologySpec:
    """Legacy FC LIF spec using the exact ``SpikingNet`` stage names.

    Deliberate checkpoint back-compat: ``SpikingNet`` stores ``_fc1``,
    ``_lif1``, ``_fc2`` and ``_lif2`` in its ``state_dict`` and its neurons
    are plain ``snn.Leaky(beta=beta)``. Reusing those names and that neuron
    keeps old checkpoints loadable into the spec-built module unchanged.
    """
    return chain(
        [
            Stage("_fc1", "linear", _linear_params(input_size, hidden)),
            Stage("_lif1", "leaky", {"beta": beta}),
            Stage("_fc2", "linear", _linear_params(hidden, num_classes)),
            Stage("_lif2", "leaky", {"beta": beta}),
        ]
    )


def fc_small(
    hidden: int = 32,
    beta: float = 0.9,
    num_classes: int = 10,
    input_size: int = 28 * 28,
) -> TopologySpec:
    """Small FC LIF spec with an explicit flatten entry stage."""
    return chain(
        [
            Stage("flatten", "flatten", {}),
            Stage("fc1", "linear", _linear_params(input_size, hidden)),
            Stage("lif1", "leaky", {"beta": beta}),
            Stage("fc2", "linear", _linear_params(hidden, num_classes)),
            Stage("lif2", "leaky", {"beta": beta}),
        ]
    )


def _conv_stages(
    in_channels: int,
    channels: int,
    num_classes: int,
    side: int,
) -> List[Stage]:
    features = channels * 2 * side * side
    return [
        Stage("conv1", "conv2d", _conv_params(in_channels, channels)),
        Stage("lif1", "leaky", {"beta": 0.9}),
        Stage("pool1", "avgpool2d", {"kernel_size": 2}),
        Stage("conv2", "conv2d", _conv_params(channels, channels * 2)),
        Stage("lif2", "leaky", {"beta": 0.9}),
        Stage("pool2", "sumpool2d", {"kernel_size": 2}),
        Stage("flatten", "flatten", {}),
        Stage("fc", "linear", _linear_params(features, num_classes)),
        Stage("out", "leaky", {"beta": 0.9}),
    ]


def conv_net(
    in_channels: int = 1,
    channels: int = 8,
    num_classes: int = 10,
    input_size: int = 28,
) -> TopologySpec:
    """Conv/pool feature extractor with a linear LIF readout."""
    side = input_size // 4
    return chain(_conv_stages(in_channels, channels, num_classes, side))


def _recurrent_stages(
    hidden: int, beta: float, num_classes: int, input_size: int
) -> List[Stage]:
    return [
        Stage("fc1", "linear", _linear_params(input_size, hidden)),
        Stage("lif1", "leaky", {"beta": beta}),
        Stage("rec", "linear", _linear_params(hidden, hidden)),
        Stage("merge", "add", {}),
        Stage("lif2", "leaky", {"beta": beta}),
        Stage("fc2", "linear", _linear_params(hidden, num_classes)),
        Stage("out", "leaky", {"beta": beta}),
    ]


def _recurrent_edges() -> List[Edge]:
    return [
        Edge("fc1", "lif1"),
        Edge("lif1", "rec"),
        Edge("lif1", "merge"),
        Edge("rec", "merge", delayed=True),
        Edge("merge", "lif2"),
        Edge("lif2", "fc2"),
        Edge("fc2", "out"),
    ]


def recurrent_net(
    hidden: int = 64,
    beta: float = 0.9,
    num_classes: int = 10,
    input_size: int = 28 * 28,
) -> TopologySpec:
    """FC LIF spec with an explicit one-step delayed feedback edge."""
    return TopologySpec(
        stages=_recurrent_stages(hidden, beta, num_classes, input_size),
        edges=_recurrent_edges(),
        input="fc1",
        output="out",
    )
