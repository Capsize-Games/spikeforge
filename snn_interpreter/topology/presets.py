"""Ready-made topology specs shared by tests, training, and validation.

A preset selects how its neuron stages are built. ``neuron`` names the
registry kind used for *every* neuron stage (default :data:`DEFAULT_NEURON`)
and ``surrogate`` an optional surrogate-gradient factory threaded into each
neuron's ``spike_grad``. Both remain ordinary preset parameters, so the
registry resolves and forwards them like any other override.

For **per-stage** heterogeneity, ``neurons`` overrides the kind of named
stages (``{"lif1": "synaptic"}``) and ``stage_params`` merges extra constructor
arguments over a stage's computed params (for example
``{"lif1": {"beta": 0.8, "reset": "zero"}}``). Leaving every override unset
reproduces the historical ``leaky``/no-surrogate specs byte for byte.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from snn_interpreter.data.image_size import SizeLike, as_size
from snn_interpreter.topology.edge import Edge
from snn_interpreter.topology.spec import TopologySpec, chain
from snn_interpreter.topology.stage import Stage

#: Neuron kind used by every preset when ``neuron`` is not overridden.
DEFAULT_NEURON = "leaky"
#: Per-stage neuron-kind overrides keyed by stage name.
NeuronMap = Optional[Mapping[str, str]]
#: Per-stage parameter overrides keyed by stage name.
ParamsMap = Optional[Mapping[str, Mapping[str, Any]]]
#: A builder for one neuron stage under the preset's overrides.
NeuronStage = Callable[[str, str, float, Optional[str]], Stage]


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
    name: str,
    neuron: str,
    beta: float,
    surrogate: Optional[str] = None,
    *,
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
) -> Stage:
    """Return a neuron stage honoring per-stage kind and param overrides."""
    kind = (neurons or {}).get(name, neuron)
    params = _neuron_params(beta, surrogate)
    params.update((stage_params or {}).get(name, {}))
    return Stage(name, kind, params)


def _stage_factory(neurons: NeuronMap, stage_params: ParamsMap) -> NeuronStage:
    """Return a neuron-stage builder bound to the preset's overrides."""
    def make(
        name: str, neuron: str, beta: float, surrogate: Optional[str] = None
    ) -> Stage:
        return _neuron_stage(
            name, neuron, beta, surrogate,
            neurons=neurons, stage_params=stage_params,
        )
    return make


def fc_legacy(
    hidden: int = 128, beta: float = 0.5,
    num_classes: int = 10, input_size: int = 28 * 28,
    neuron: str = DEFAULT_NEURON, surrogate: Optional[str] = None,
    neurons: NeuronMap = None, stage_params: ParamsMap = None,
) -> TopologySpec:
    """Legacy FC LIF spec using the exact ``SpikingNet`` stage names.

    The registry renders this through ``SpikingNet`` only when no override is
    present, preserving its state-dict keys.
    """
    stage = _stage_factory(neurons, stage_params)
    return chain(
        [
            Stage("_fc1", "linear", _linear_params(input_size, hidden)),
            stage("_lif1", neuron, beta, surrogate),
            Stage("_fc2", "linear", _linear_params(hidden, num_classes)),
            stage("_lif2", neuron, beta, surrogate),
        ]
    )


def fc_small(
    hidden: int = 32,
    beta: float = 0.9,
    num_classes: int = 10,
    input_size: int = 28 * 28,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
) -> TopologySpec:
    """Small FC LIF spec with an explicit flatten entry stage."""
    stage = _stage_factory(neurons, stage_params)
    return chain(
        [
            Stage("flatten", "flatten", {}),
            Stage("fc1", "linear", _linear_params(input_size, hidden)),
            stage("lif1", neuron, beta, surrogate),
            Stage("fc2", "linear", _linear_params(hidden, num_classes)),
            stage("lif2", neuron, beta, surrogate),
        ]
    )


#: Smallest sensor side the two 2x2 pools can downsample to a live feature.
_MIN_SIDE = 4


def _pooled_size(input_size: SizeLike) -> Tuple[int, int]:
    """Return the pooled ``(H, W)`` conv_net keeps, or raise if too small."""
    height, width = as_size(input_size)
    if height < _MIN_SIDE or width < _MIN_SIDE:
        raise ValueError(
            f"conv_net needs each input side >= {_MIN_SIDE} for its two "
            f"2x2 pools, got {height}x{width}"
        )
    return (height // 4, width // 4)


def _conv_stages(
    in_channels: int,
    channels: int,
    num_classes: int,
    size: Tuple[int, int],
    neuron: str,
    surrogate: Optional[str],
    stage: NeuronStage,
) -> List[Stage]:
    height, width = size
    features = channels * 2 * height * width
    return [
        Stage("conv1", "conv2d", _conv_params(in_channels, channels)),
        stage("lif1", neuron, 0.9, surrogate),
        Stage("pool1", "avgpool2d", {"kernel_size": 2}),
        Stage("conv2", "conv2d", _conv_params(channels, channels * 2)),
        stage("lif2", neuron, 0.9, surrogate),
        Stage("pool2", "sumpool2d", {"kernel_size": 2}),
        Stage("flatten", "flatten", {}),
        Stage("fc", "linear", _linear_params(features, num_classes)),
        stage("out", neuron, 0.9, surrogate),
    ]


def conv_net(
    in_channels: int = 1,
    channels: int = 8,
    num_classes: int = 10,
    input_size: SizeLike = 28,
    neuron: str = DEFAULT_NEURON,
    surrogate: Optional[str] = None,
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
) -> TopologySpec:
    """Conv/pool feature extractor with a linear LIF readout.

    ``input_size`` is the sensor geometry: an ``int`` square side (unchanged)
    or an explicit ``(H, W)`` pair. The two 2x2 pools leave ``H/4 * W/4``
    features per doubled channel, so each side must be at least 4.
    """
    size = _pooled_size(input_size)
    stage = _stage_factory(neurons, stage_params)
    stages = _conv_stages(
        in_channels, channels, num_classes, size, neuron, surrogate, stage
    )
    return chain(stages)


def _recurrent_stages(
    hidden: int, beta: float, num_classes: int, input_size: int,
    neuron: str, surrogate: Optional[str], stage: NeuronStage,
) -> List[Stage]:
    return [
        Stage("fc1", "linear", _linear_params(input_size, hidden)),
        stage("lif1", neuron, beta, surrogate),
        Stage("rec", "linear", _linear_params(hidden, hidden)),
        Stage("merge", "add", {}),
        stage("lif2", neuron, beta, surrogate),
        Stage("fc2", "linear", _linear_params(hidden, num_classes)),
        stage("out", neuron, beta, surrogate),
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
    neurons: NeuronMap = None,
    stage_params: ParamsMap = None,
) -> TopologySpec:
    """FC LIF spec with an explicit one-step delayed feedback edge."""
    stage = _stage_factory(neurons, stage_params)
    stages = _recurrent_stages(
        hidden, beta, num_classes, input_size, neuron, surrogate, stage
    )
    return TopologySpec(
        stages=stages,
        edges=_recurrent_edges(),
        input="fc1",
        output="out",
    )
