"""Ready-made topology specs shared by tests, training, and validation.

Every preset accepts two overrides that select how its neuron stages are
built. ``neuron`` names the registry kind used for *all* neuron stages
(default :data:`DEFAULT_NEURON`), and ``surrogate`` names an optional
surrogate-gradient factory threaded into each neuron's ``spike_grad``.
Both are ordinary preset parameters, so the registry resolves and forwards
them like any other override; leaving them at their defaults reproduces the
historical ``leaky``/no-surrogate specs exactly.
"""

from typing import Any, Dict, List, Optional

from snn_interpreter.topology.edge import Edge
from snn_interpreter.topology.spec import TopologySpec, chain
from snn_interpreter.topology.stage import Stage

#: Neuron kind used by every preset when ``neuron`` is not overridden.
DEFAULT_NEURON = "leaky"


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


def _neuron_params(
    beta: float, surrogate: Optional[str] = None
) -> Dict[str, Any]:
    """Return a neuron stage's params, omitting an unset surrogate."""
    params: Dict[str, Any] = {"beta": beta}
    if surrogate is not None:
        params["surrogate"] = surrogate
    return params


def _neuron_stage(
    name: str, neuron: str, beta: float, surrogate: Optional[str]
) -> Stage:
    """Return a neuron stage of kind ``neuron`` for a preset."""
    return Stage(name, neuron, _neuron_params(beta, surrogate))


def fc_legacy(
    hidden: int = 128, beta: float = 0.5,
    num_classes: int = 10, input_size: int = 28 * 28,
    neuron: str = DEFAULT_NEURON, surrogate: Optional[str] = None,
) -> TopologySpec:
    """Legacy FC LIF spec using the exact ``SpikingNet`` stage names.

    The registry renders this through ``SpikingNet`` only for the default
    ``leaky`` neuron with no surrogate, preserving its state-dict keys.
    """
    return chain(
        [
            Stage("_fc1", "linear", _linear_params(input_size, hidden)),
            _neuron_stage("_lif1", neuron, beta, surrogate),
            Stage("_fc2", "linear", _linear_params(hidden, num_classes)),
            _neuron_stage("_lif2", neuron, beta, surrogate),
        ]
    )


def fc_small(
    hidden: int = 32,
    beta: float = 0.9,
    num_classes: int = 10,
    input_size: int = 28 * 28,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
) -> TopologySpec:
    """Small FC LIF spec with an explicit flatten entry stage."""
    return chain(
        [
            Stage("flatten", "flatten", {}),
            Stage("fc1", "linear", _linear_params(input_size, hidden)),
            _neuron_stage("lif1", neuron, beta, surrogate),
            Stage("fc2", "linear", _linear_params(hidden, num_classes)),
            _neuron_stage("lif2", neuron, beta, surrogate),
        ]
    )


def _conv_stages(
    in_channels: int,
    channels: int,
    num_classes: int,
    side: int,
    neuron: str,
    surrogate: Optional[str],
) -> List[Stage]:
    features = channels * 2 * side * side
    return [
        Stage("conv1", "conv2d", _conv_params(in_channels, channels)),
        _neuron_stage("lif1", neuron, 0.9, surrogate),
        Stage("pool1", "avgpool2d", {"kernel_size": 2}),
        Stage("conv2", "conv2d", _conv_params(channels, channels * 2)),
        _neuron_stage("lif2", neuron, 0.9, surrogate),
        Stage("pool2", "sumpool2d", {"kernel_size": 2}),
        Stage("flatten", "flatten", {}),
        Stage("fc", "linear", _linear_params(features, num_classes)),
        _neuron_stage("out", neuron, 0.9, surrogate),
    ]


def conv_net(
    in_channels: int = 1,
    channels: int = 8,
    num_classes: int = 10,
    input_size: int = 28,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
) -> TopologySpec:
    """Conv/pool feature extractor with a linear LIF readout."""
    side = input_size // 4
    stages = _conv_stages(
        in_channels, channels, num_classes, side, neuron, surrogate
    )
    return chain(stages)


def _recurrent_stages(
    hidden: int, beta: float, num_classes: int, input_size: int,
    neuron: str, surrogate: Optional[str],
) -> List[Stage]:
    return [
        Stage("fc1", "linear", _linear_params(input_size, hidden)),
        _neuron_stage("lif1", neuron, beta, surrogate),
        Stage("rec", "linear", _linear_params(hidden, hidden)),
        Stage("merge", "add", {}),
        _neuron_stage("lif2", neuron, beta, surrogate),
        Stage("fc2", "linear", _linear_params(hidden, num_classes)),
        _neuron_stage("out", neuron, beta, surrogate),
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
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
) -> TopologySpec:
    """FC LIF spec with an explicit one-step delayed feedback edge."""
    stages = _recurrent_stages(
        hidden, beta, num_classes, input_size, neuron, surrogate
    )
    return TopologySpec(
        stages=stages,
        edges=_recurrent_edges(),
        input="fc1",
        output="out",
    )
