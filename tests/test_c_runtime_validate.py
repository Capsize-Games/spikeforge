"""C-runtime validation of hand-built export summaries."""

import json
from typing import Any, Dict, List, Tuple

import pytest

from spikeforge_targets.c_runtime import TARGET_NAME
from spikeforge_targets.c_runtime_errors import CRuntimeUnsupportedError
from spikeforge_targets.c_runtime_validate import (
    check_for_c_runtime,
    validate_for_c_runtime,
)

_Node = Dict[str, Any]
_Edge = Dict[str, str]


def _node(name: str, kind: str, params: _Node) -> _Node:
    """Return one summary node record."""
    return {"name": name, "kind": kind, "params": params}


def _edge(source: str, target: str) -> _Edge:
    """Return one summary edge record."""
    return {"source": source, "target": target}


def _affine(name: str) -> _Node:
    """Return a conforming 2x3 Affine node record."""
    return _node(
        name,
        "Affine",
        {"weight": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], "bias": [0.5, 0.5]},
    )


def _chain(*nodes: _Node) -> Dict[str, Any]:
    """Return a summary chaining input, ``nodes``, and output."""
    chain = [_node("input", "Input", {})]
    chain.extend(nodes)
    chain.append(_node("output", "Output", {}))
    edges = [
        _edge(chain[index]["name"], chain[index + 1]["name"])
        for index in range(len(chain) - 1)
    ]
    return {"nodes": chain, "edges": edges}


def _pairs(report: Any) -> List[Tuple[str, str]]:
    """Return the ``(name, kind)`` pairs of a report's violations."""
    return [(item.name, item.kind) for item in report.unsupported]


def test_minimal_affine_chain_validates_clean() -> None:
    """A well-formed chain passes with every node supported."""
    report = validate_for_c_runtime(_chain(_affine("fc")))
    assert report.target == TARGET_NAME
    assert report.passed() is True
    assert report.unsupported == ()
    assert json.dumps(report.to_dict())
    assert check_for_c_runtime(_chain(_affine("fc"))) is None


@pytest.mark.parametrize(
    "kind",
    ["Linear", "Conv1d", "LIF", "IF", "CubaLIF"],
)
def test_known_gap_kinds_report_the_exact_node(kind: str) -> None:
    """Each C-rejected kind is reported by node name and kind."""
    summary = _chain(_node("gap", kind, {}))
    report = validate_for_c_runtime(summary)
    assert _pairs(report) == [("gap", kind)]
    assert kind in report.unsupported[0].reason


def test_unknown_kind_is_reported_not_crashed() -> None:
    """A kind outside every table fails with a named violation."""
    report = validate_for_c_runtime(_chain(_node("mystery", "Nope", {})))
    assert _pairs(report) == [("mystery", "Nope")]


def test_affine_without_bias_is_rejected() -> None:
    """Affine needs its bias: the C loader fails the node without one."""
    node = _node("fc", "Affine", {"weight": [[1.0]], "bias": None})
    report = validate_for_c_runtime(_chain(node))
    assert _pairs(report) == [("fc", "Affine")]
    assert "bias" in report.unsupported[0].reason


def test_ragged_affine_weight_is_rejected() -> None:
    """A non-rectangular weight would misread in the C loader."""
    node = _node("fc", "Affine", {"weight": [[1.0], [2.0, 3.0]]})
    report = validate_for_c_runtime(_chain(node))
    assert _pairs(report) == [("fc", "Affine")]


def test_valid_quantized_envelope_is_accepted() -> None:
    """A well-formed int8 envelope matches the C weight_view parsing."""
    node = _node(
        "fc",
        "Affine",
        {
            "weight": {
                "data": [[1, 2], [3, 4]],
                "dtype": "int8",
                "scale": 0.5,
                "zero_point": 0,
            },
            "bias": [0.5, 0.5],
        },
    )
    assert validate_for_c_runtime(_chain(node)).passed() is True


def test_envelope_with_bad_dtype_is_rejected() -> None:
    """Only int8/uint8 envelopes reach the C dequantizer."""
    node = _node(
        "fc",
        "Affine",
        {
            "weight": {
                "data": [[1, 2], [3, 4]],
                "dtype": "int4",
                "scale": 0.5,
                "zero_point": 0,
            },
            "bias": [0.5, 0.5],
        },
    )
    assert _pairs(validate_for_c_runtime(_chain(node))) == [("fc", "Affine")]


def _conv(params: _Node) -> _Node:
    """Return a Conv2d node record with conforming defaults overridden."""
    full: _Node = {
        "weight": [[[[1.0]]]],
        "bias": [0.5],
        "stride": [1, 1],
        "padding": [0, 0],
        "dilation": [1, 1],
        "groups": 1,
    }
    full.update(params)
    return _node("conv", "Conv2d", full)


def test_conforming_conv_validates_clean() -> None:
    """A dense conv with pairs and 4D weights passes."""
    assert validate_for_c_runtime(_chain(_conv({}))).passed() is True


@pytest.mark.parametrize(
    "params",
    [
        {"weight": [[1.0]]},
        {"bias": [0.5, 0.5]},
        {"stride": 1},
        {"stride": [1, 0]},
        {"padding": "same"},
        {"dilation": [1, 1, 1]},
        {"groups": 2},
    ],
)
def test_malformed_conv_is_rejected(params: _Node) -> None:
    """Bad conv layouts, scalar grids, and groups fail by node name."""
    report = validate_for_c_runtime(_chain(_conv(params)))
    assert _pairs(report) == [("conv", "Conv2d")]


def test_conv_without_bias_is_accepted() -> None:
    """A null conv bias means no bias to the C loader."""
    assert validate_for_c_runtime(_chain(_conv({"bias": None}))).passed()


def test_pool_with_zero_stride_is_rejected() -> None:
    """A zero pool stride would divide by zero in the C pooling op."""
    node = _node(
        "pool",
        "AvgPool2d",
        {"kernel_size": [2, 2], "stride": [2, 0], "padding": [0, 0]},
    )
    assert _pairs(validate_for_c_runtime(_chain(node))) == [
        ("pool", "AvgPool2d")
    ]


def test_partial_flatten_is_rejected() -> None:
    """Flatten must cover every non-batch dim like the C op does."""
    node = _node("fl", "Flatten", {"start_dim": 2, "end_dim": -1})
    report = validate_for_c_runtime(_chain(node))
    assert _pairs(report) == [("fl", "Flatten")]
    good = _node("fl", "Flatten", {"start_dim": 1, "end_dim": -1})
    assert validate_for_c_runtime(_chain(good)).passed() is True


def test_two_step_delay_is_rejected() -> None:
    """The C runtime hardcodes a one-step delay."""
    node = _node("wait", "Delay", {"delay": 2.0})
    assert _pairs(validate_for_c_runtime(_chain(node))) == [("wait", "Delay")]
    good = _node("wait", "Delay", {"delay": 1.0})
    assert validate_for_c_runtime(_chain(good)).passed() is True


def test_li_with_non_positive_tau_is_rejected() -> None:
    """The C LI op rejects tau <= 0."""
    node = _node("mem", "LI", {"tau": 0.0, "r": 1.0, "v_leak": 0.0})
    assert _pairs(validate_for_c_runtime(_chain(node))) == [("mem", "LI")]
    good = _node("mem", "LI", {"tau": 9.0, "r": 1.0, "v_leak": 0.0})
    assert validate_for_c_runtime(_chain(good)).passed() is True


def test_non_numeric_threshold_is_rejected() -> None:
    """A threshold the C loader cannot read fails by node name."""
    node = _node("spike", "Threshold", {"threshold": "high"})
    report = validate_for_c_runtime(_chain(node))
    assert _pairs(report) == [("spike", "Threshold")]


def test_edge_to_unknown_node_is_rejected() -> None:
    """An edge naming an absent node fails the C edge load."""
    summary = _chain(_affine("fc"))
    summary["edges"].append(_edge("fc", "ghost"))
    report = validate_for_c_runtime(summary)
    assert ("fc->ghost", "edge") in _pairs(report)


def test_starved_node_is_rejected() -> None:
    """A node with no incoming edge has nothing for the C sum."""
    summary = _chain(_affine("fc"))
    summary["nodes"].append(_node("lost", "Scale", {"scale": 1.0}))
    report = validate_for_c_runtime(summary)
    assert ("lost", "Scale") in _pairs(report)


def test_missing_output_is_rejected() -> None:
    """The C runtime runs a single input to a readout."""
    summary = {
        "nodes": [_node("input", "Input", {}), _affine("fc")],
        "edges": [_edge("input", "fc")],
    }
    report = validate_for_c_runtime(summary)
    assert ("graph", "graph") in _pairs(report)


def test_unbroken_cycle_is_rejected() -> None:
    """A cycle with no Delay cannot order in the C runtime."""
    summary = {
        "nodes": [
            _node("input", "Input", {}),
            _node("a", "Scale", {"scale": 1.0}),
            _node("b", "Scale", {"scale": 1.0}),
            _node("output", "Output", {}),
        ],
        "edges": [
            _edge("input", "a"),
            _edge("a", "b"),
            _edge("b", "a"),
            _edge("b", "output"),
        ],
    }
    report = validate_for_c_runtime(summary)
    assert ("graph", "graph") in _pairs(report)


def test_delay_broken_cycle_validates_clean() -> None:
    """A Delay on the feedback edge orders the C runtime's steps."""
    summary = {
        "nodes": [
            _node("input", "Input", {}),
            _node("a", "Scale", {"scale": 1.0}),
            _node("wait", "Delay", {"delay": 1.0}),
            _node("output", "Output", {}),
        ],
        "edges": [
            _edge("input", "a"),
            _edge("a", "wait"),
            _edge("wait", "a"),
            _edge("a", "output"),
        ],
    }
    assert validate_for_c_runtime(summary).passed() is True


def test_duplicate_names_are_rejected() -> None:
    """Repeated names misroute C edges to the first match."""
    summary = _chain(_affine("fc"))
    summary["nodes"].append(_affine("fc"))
    report = validate_for_c_runtime(summary)
    assert ("fc", "Affine") in _pairs(report)


def test_check_raises_the_typed_error() -> None:
    """The gate raises naming names and kinds, passing clean graphs."""
    summary = _chain(_node("gap", "LIF", {}))
    with pytest.raises(CRuntimeUnsupportedError) as excinfo:
        check_for_c_runtime(summary)
    assert excinfo.value.names == ("gap",)
    assert excinfo.value.kinds == ("LIF",)


def test_malformed_summary_raises_value_error() -> None:
    """Inputs that are not summaries fail fast instead of misreading."""
    with pytest.raises(ValueError, match="summary"):
        validate_for_c_runtime({"nodes": []})
    with pytest.raises(ValueError, match="summary"):
        validate_for_c_runtime({"nodes": [], "edges": {}})
    with pytest.raises(ValueError, match="TopologySpec"):
        validate_for_c_runtime(object())
