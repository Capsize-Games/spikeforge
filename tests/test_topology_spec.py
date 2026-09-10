"""Tests for topology graphs, builders, validation, and round-trips."""

import pytest

from snn_interpreter.topology.edge import Edge
from snn_interpreter.topology.spec import (
    TopologySpec,
    chain,
    multi_branch,
    recurrent,
    residual,
)
from snn_interpreter.topology.stage import Stage

_A = Stage("a", "linear", {"in_features": 2, "out_features": 2})
_B = Stage("b", "linear", {"in_features": 2, "out_features": 2})
_C = Stage("c", "linear", {"in_features": 2, "out_features": 2})
_MERGE = Stage("m", "add", {})


def _pairs(spec: TopologySpec) -> set:
    return {(edge.source, edge.target) for edge in spec.edges}


def test_chain_links_stages_head_to_tail() -> None:
    """A chain links consecutive stages and sets endpoints."""
    spec = chain([_A, _B, _C])
    assert _pairs(spec) == {("a", "b"), ("b", "c")}
    assert spec.input == "a"
    assert spec.output == "c"


def test_residual_adds_skip_edge() -> None:
    """The residual helper adds a weight-1 skip edge."""
    spec = residual([_A, _B, _MERGE], "a", "m")
    assert Edge("a", "m") in spec.edges
    assert _pairs(spec) == {("a", "b"), ("b", "m"), ("a", "m")}


def test_multi_branch_fans_out_and_merges() -> None:
    """Every branch starts at the input stage and ends at the merge."""
    spec = multi_branch(_A, [[_B], [_C]], _MERGE)
    assert _pairs(spec) == {
        ("a", "b"),
        ("a", "c"),
        ("b", "m"),
        ("c", "m"),
    }
    assert spec.input == "a"
    assert spec.output == "m"


def test_recurrent_adds_delayed_feedback_edge() -> None:
    """The recurrent helper marks its feedback edge as delayed."""
    spec = recurrent([_A, _B], "b", "a")
    assert Edge("b", "a", delayed=True) in spec.edges


def test_recurrent_spec_validates() -> None:
    """A delayed feedback edge is not treated as a forward cycle."""
    recurrent([_A, _B], "b", "a").validate()


def test_validate_rejects_unknown_reference() -> None:
    """An edge naming a missing stage is rejected."""
    spec = TopologySpec(
        stages=[_A],
        edges=[Edge("a", "ghost")],
        input="a",
        output="a",
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_validate_rejects_cycle() -> None:
    """A forward cycle is rejected."""
    spec = TopologySpec(
        stages=[_A, _B],
        edges=[Edge("a", "b"), Edge("b", "a")],
        input="a",
        output="b",
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_validate_rejects_unreachable_stage() -> None:
    """A stage with no path from the input is rejected."""
    spec = TopologySpec(
        stages=[_A, _B, _C],
        edges=[Edge("a", "b")],
        input="a",
        output="b",
    )
    with pytest.raises(ValueError):
        spec.validate()


def test_dict_round_trip() -> None:
    """to_dict/from_dict reconstruct an equal spec."""
    spec = residual([_A, _B, _MERGE], "a", "m")
    assert TopologySpec.from_dict(spec.to_dict()) == spec


def test_from_dict_preserves_delay_flag() -> None:
    """The delayed flag survives serialization."""
    spec = recurrent([_A, _B], "b", "a")
    restored = TopologySpec.from_dict(spec.to_dict())
    assert any(edge.delayed for edge in restored.edges)
