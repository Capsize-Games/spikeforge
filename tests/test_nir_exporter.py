"""Export presets and merge topologies to NIR graphs and summaries."""

import json
from typing import Any, List, Set, Tuple

import numpy as np
import pytest

from snn_interpreter.nir_bridge.exporter import graph_summary, to_nir
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module
from snn_interpreter.topology.spec import (
    TopologySpec,
    multi_branch,
    residual,
)
from snn_interpreter.topology.stage import Stage

pytest.importorskip("nir")

_PRESETS = [
    presets.fc_legacy(hidden=4, beta=0.5, num_classes=2, input_size=3),
    presets.fc_small(hidden=4, beta=0.9, num_classes=2, input_size=3),
    presets.conv_net(in_channels=1, channels=2, num_classes=3, input_size=8),
    presets.recurrent_net(hidden=4, beta=0.9, num_classes=2, input_size=3),
]
_IDS = ["fc_legacy", "fc_small", "conv_net", "recurrent_net"]


def _export(spec: TopologySpec) -> Any:
    """Build ``spec`` and export it together with its weights."""
    return to_nir(spec, build_module(spec))


def _kinds(graph: Any) -> List[str]:
    """Return the sorted node kind names of ``graph``."""
    return sorted(type(node).__name__ for node in graph.nodes.values())


def _edges(graph: Any) -> Set[Tuple[str, str]]:
    """Return the graph's edges as a set of name pairs."""
    return set(graph.edges)


def _linear(name: str) -> Stage:
    """Return a small 3x3 linear stage."""
    return Stage(name, "linear", {"in_features": 3, "out_features": 3})


_FC_LEGACY_KINDS = sorted(
    [
        "Input",
        "Affine",
        "LI",
        "Threshold",
        "Delay",
        "Scale",
        "Affine",
        "LI",
        "Threshold",
        "Delay",
        "Scale",
        "Output",
    ]
)
_FC_LEGACY_EDGES = {
    ("input", "_fc1"),
    ("_fc1", "_lif1__mem"),
    ("_lif1", "_fc2"),
    ("_fc2", "_lif2__mem"),
    ("_lif2", "output"),
    ("_lif1__mem", "_lif1"),
    ("_lif1", "_lif1__reset_delay"),
    ("_lif1__reset_delay", "_lif1__reset_scale"),
    ("_lif1__reset_scale", "_lif1__mem"),
    ("_lif2__mem", "_lif2"),
    ("_lif2", "_lif2__reset_delay"),
    ("_lif2__reset_delay", "_lif2__reset_scale"),
    ("_lif2__reset_scale", "_lif2__mem"),
}


def test_fc_legacy_export_structure() -> None:
    """fc_legacy exports Affine stages feeding split LI/threshold neurons."""
    graph = _export(
        presets.fc_legacy(hidden=4, beta=0.5, num_classes=2, input_size=3)
    )
    assert _kinds(graph) == _FC_LEGACY_KINDS
    assert _edges(graph) == _FC_LEGACY_EDGES


def test_fc_small_export_includes_flatten() -> None:
    """fc_small starts with a Flatten node feeding the first Affine."""
    graph = _export(
        presets.fc_small(hidden=4, beta=0.9, num_classes=2, input_size=3)
    )
    assert "Flatten" in _kinds(graph)
    assert ("flatten", "fc1") in _edges(graph)


_CONV_KINDS = {
    "Input",
    "Conv2d",
    "AvgPool2d",
    "SumPool2d",
    "Flatten",
    "Affine",
    "LI",
    "Threshold",
    "Delay",
    "Scale",
    "Output",
}
_CONV_EDGES = {("pool1", "conv2"), ("pool2", "flatten")}


def test_conv_net_export_structure() -> None:
    """conv_net exports conv, pooling, flatten and split readout nodes."""
    graph = _export(
        presets.conv_net(
            in_channels=1, channels=2, num_classes=3, input_size=8
        )
    )
    assert set(_kinds(graph)) >= _CONV_KINDS
    assert _edges(graph) >= _CONV_EDGES


def test_recurrent_net_export_has_delay_node() -> None:
    """The delayed feedback edge becomes an explicit Delay node."""
    graph = _export(
        presets.recurrent_net(
            hidden=4, beta=0.9, num_classes=2, input_size=3
        )
    )
    assert "Delay" in _kinds(graph)
    edges = _edges(graph)
    assert ("rec", "rec__delay__lif2__mem") in edges
    assert ("rec__delay__lif2__mem", "lif2__mem") in edges
    assert ("lif1", "lif2__mem") in edges


def test_residual_converges_on_output() -> None:
    """A residual skip and the branch both converge on the output."""
    stages = [Stage("src", "add", {}), _linear("branch")]
    stages.append(Stage("merge", "add", {}))
    spec = residual(stages, "src", "merge")
    edges = _edges(_export(spec))
    assert ("input", "branch") in edges
    assert ("branch", "output") in edges
    assert ("input", "output") in edges


def test_multi_branch_converges_on_output() -> None:
    """Both branches fan out from the input and merge at the output."""
    spec = multi_branch(
        Stage("src", "add", {}),
        [[_linear("left")], [_linear("right")]],
        Stage("merge", "add", {}),
    )
    edges = _edges(_export(spec))
    assert ("input", "left") in edges
    assert ("input", "right") in edges
    assert ("left", "output") in edges
    assert ("right", "output") in edges


def test_conv_export_copies_module_weights() -> None:
    """Conv and linear nodes carry the module's learned tensors."""
    spec = presets.conv_net(
        in_channels=1, channels=4, num_classes=5, input_size=8
    )
    module = build_module(spec)
    graph = to_nir(spec, module)
    conv = module.get_submodule("conv1").weight.detach().numpy()
    fc = module.get_submodule("fc").weight.detach().numpy()
    assert np.allclose(graph.nodes["conv1"].weight, conv)
    assert np.allclose(graph.nodes["fc"].weight, fc)


@pytest.mark.parametrize("spec", _PRESETS, ids=_IDS)
def test_graph_summary_is_json_serialisable(spec: TopologySpec) -> None:
    """Every preset yields a JSON-able summary from the spec alone."""
    summary = graph_summary(spec)
    assert json.dumps(summary)
    kinds = {node["kind"] for node in summary["nodes"]}
    assert {"Input", "Output"} <= kinds


@pytest.mark.parametrize("spec", _PRESETS, ids=_IDS)
def test_exported_graph_summary_is_json_serialisable(
    spec: TopologySpec,
) -> None:
    """The exported graph also yields a JSON-able summary."""
    assert json.dumps(graph_summary(_export(spec)))
