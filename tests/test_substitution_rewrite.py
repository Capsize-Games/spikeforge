"""Substitution rewrite correctness, buckets, and drift detection."""

import json
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import api
from spikeforge.nir_bridge.exporter import to_nir
from spikeforge.topology import presets
from spikeforge_targets.rewrite import rewrite
from spikeforge_targets.rewrite_drift import rewrite_drift
from spikeforge_targets.target_spec import TargetSpec

pytest.importorskip("nir")

_CONV = presets.conv_net(
    in_channels=1, channels=2, num_classes=3, input_size=8
)


def _graph(nodes: Dict[str, Any], edges: List[Tuple[str, str]]) -> Any:
    """Build an unchecked NIR graph from named nodes and edges."""
    return api.node_class("NIRGraph")(nodes, edges, type_check=False)


def _io() -> Tuple[Any, Any]:
    """Return fresh ``Input`` and ``Output`` nodes."""
    return (
        api.node_class("Input")({"input": None}),
        api.node_class("Output")({"output": None}),
    )


def _if_graph() -> Any:
    """Return an ``Input -> IF -> Output`` graph with unit resistance."""
    source, sink = _io()
    node = api.node_class("IF")(
        np.array(1.0, np.float32),
        np.array(1.0, np.float32),
        np.array(0.0, np.float32),
    )
    return _graph({"input": source, "a": node, "output": sink},
                  [("input", "a"), ("a", "output")])


def _lif_graph(tau: float, threshold: float = 1.0) -> Any:
    """Return an ``Input -> LIF -> Output`` graph with distinct ``tau``."""
    source, sink = _io()
    node = api.node_class("LIF")(
        np.array(tau, np.float32),
        np.array(1.0, np.float32),
        np.array(0.0, np.float32),
        np.array(threshold, np.float32),
        np.array(0.0, np.float32),
    )
    return _graph({"input": source, "a": node, "output": sink},
                  [("input", "a"), ("a", "output")])


def test_lava_rewrite_expands_avgpool_into_sumpool_scale() -> None:
    """AvgPool2d becomes SumPool2d plus a 1/4 Scale, edges rewired."""
    graph = to_nir(_CONV)
    result = rewrite(graph, "lava_loihi2")
    applied = {item["node"]: item for item in result.report.applied}
    assert applied["pool1"]["from"] == "AvgPool2d"
    assert applied["pool1"]["to"] == "SumPool2d"
    assert applied["pool1"]["detail"] == "1/4"
    assert result.report.ready() is True
    kinds = {
        name: type(node).__name__
        for name, node in result.graph.nodes.items()
    }
    assert kinds["pool1"] == "SumPool2d"
    assert kinds["pool1__scale"] == "Scale"
    assert ("pool1", "pool1__scale") in result.graph.edges
    assert ("pool1__scale", "conv2") in result.graph.edges


def test_lava_rewrite_is_drift_free() -> None:
    """The sum-pool plus scale rewrite reproduces the average exactly."""
    graph = to_nir(_CONV)
    torch.manual_seed(0)
    spikes = torch.rand(4, 2, 1, 8, 8)
    result = rewrite(graph, "lava_loihi2", spikes)
    drift = result.report.drift
    assert drift["within_tolerance"] is True
    assert drift["readout"]["max_abs"] == 0.0


def test_norse_rewrite_converts_if_to_leak_free_lif() -> None:
    """The declared IF substitution is applied and its drift is reported."""
    torch.manual_seed(0)
    spikes = torch.rand(6, 2, 4)
    result = rewrite(_if_graph(), "norse", spikes)
    assert [item["node"] for item in result.report.applied] == ["a"]
    assert result.report.applied[0]["detail"] == "beta=0"
    assert type(result.graph.nodes["a"]).__name__ == "LIF"
    assert result.report.drift["within_tolerance"] is True


def test_unfixable_primitive_is_listed_not_dropped() -> None:
    """A primitive with no substitution is reported and kept in the graph."""
    source, sink = _io()
    weight = np.zeros((1, 1, 3, 3), np.float32)
    conv = api.node_class("Conv2d")(
        None, weight, 1, 0, 1, 1, np.zeros(1, np.float32)
    )
    graph = _graph({"input": source, "conv": conv, "output": sink},
                   [("input", "conv"), ("conv", "output")])
    result = rewrite(graph, "xylo")
    assert result.report.applied == ()
    assert result.report.ready() is False
    unfixable = {item["node"]: item for item in result.report.unfixable}
    assert unfixable["conv"]["primitive"] == "Conv2d"
    assert unfixable["conv"]["reason"] == "no substitution declared"
    assert result.graph.nodes["conv"] is conv


def test_declared_but_unexecutable_substitution_is_skipped() -> None:
    """A declared substitution with no rule lands in the skipped bucket."""
    spec = TargetSpec(
        name="synthetic",
        kind="simulator",
        description="Declares a substitution with no executable rule.",
        extra=None,
        supported=frozenset({"Input", "Output"}),
        substitutions={"LIF": "IF"},
    )
    result = rewrite(_lif_graph(2.0), spec)
    assert result.report.applied == ()
    assert result.report.ready() is False
    skipped = result.report.skipped[0]
    assert skipped["from"] == "LIF"
    assert skipped["to"] == "IF"
    assert "no executable rule" in skipped["reason"]


def test_rewrite_report_is_json_serialisable() -> None:
    """The report serialises and carries the bucket counts."""
    result = rewrite(to_nir(_CONV), "lava_loihi2")
    payload = json.loads(json.dumps(result.report.to_dict()))
    assert payload["counts"]["applied"] == 1
    assert payload["rewritten"] is True
    assert payload["ready"] is True


def test_drift_check_detects_a_regression() -> None:
    """Two different LIF constants produce a named drift regression."""
    torch.manual_seed(0)
    spikes = torch.rand(4, 2, 3)
    low = _lif_graph(2.0, threshold=0.1)
    high = _lif_graph(9.0, threshold=0.1)
    drift = rewrite_drift(low, high, spikes)
    assert drift["within_tolerance"] is False
    assert drift["readout"]["max_abs"] > 0.0


def test_rewrite_does_not_mutate_the_original_graph() -> None:
    """The source graph keeps its original nodes and edges."""
    graph = to_nir(_CONV)
    before = list(graph.nodes)
    rewrite(graph, "lava_loihi2")
    assert list(graph.nodes) == before
    assert "pool1__scale" not in graph.nodes
