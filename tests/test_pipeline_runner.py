"""Executing a pipeline graph over real checkpoints."""

from typing import Any, List

import pytest
import torch

from spikeforge.network import model_store
from spikeforge.serving.bundle import DeploymentBundle, build
from spikeforge.training.training_engine import TrainingEngine
from spikeforge_serve.pipeline import PipelineEdge, PipelineGraph, PipelineNode
from spikeforge_serve.pipeline_runner import (
    PipelineRunError,
    _extract,
    run_pipeline,
)


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads/writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


def _save_checkpoint(name: str, num_classes: int) -> None:
    """Save a tiny, deterministic fc_small checkpoint (784-feature input)."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=3,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 4, "num_classes": num_classes},
    )
    engine.save(name)


def _bundle_for(name: str) -> DeploymentBundle:
    """Build an in-memory bundle for a saved checkpoint (no disk write)."""
    return build(name)


def _raw_image() -> List[List[List[float]]]:
    """Return a flat black [1, 28, 28] image, the shape a raw frame needs."""
    return [[[0.0] * 28 for _ in range(28)]]


# --- _extract: the fixed shaping modes, tested in isolation --------------


def test_extract_mean_logits_returns_the_full_vector() -> None:
    """mean_logits passes the whole readout vector through unchanged."""
    edge = PipelineEdge("e1", "a", "b", "mean_logits")
    upstream = {"mean_logits": {"values": [[0.1, 0.2, 0.7]]}, "predicted": 2}
    assert _extract(edge, upstream, target_num_classes=3) == [0.1, 0.2, 0.7]


def test_extract_predicted_class_returns_a_single_scalar() -> None:
    """predicted_class reduces the result to one number."""
    edge = PipelineEdge("e1", "a", "b", "predicted_class")
    upstream = {"mean_logits": {"values": [[0.1, 0.9]]}, "predicted": 1}
    assert _extract(edge, upstream, target_num_classes=1) == [1.0]


def test_extract_one_hot_sizes_to_the_target_num_classes() -> None:
    """one_hot sizes its vector to the *target* node's own num_classes."""
    edge = PipelineEdge("e1", "a", "b", "one_hot")
    upstream = {"mean_logits": {"values": [[0.1, 0.2, 0.7]]}, "predicted": 2}
    assert _extract(edge, upstream, target_num_classes=4) == [
        0.0, 0.0, 1.0, 0.0,
    ]


# --- run_pipeline: real end-to-end execution ------------------------------


def test_two_node_pipeline_runs_source_then_target() -> None:
    """A -> B: B's input is A's mean_logits vector, sized to B's 784 input.

    A's num_classes is set to 784 purely so its mean_logits vector matches
    B's (default) input size -- the point under test is the chaining, not
    a realistic class count.
    """
    _save_checkpoint("digits", num_classes=784)
    _save_checkpoint("downstream", num_classes=2)
    graph = PipelineGraph(
        name="g",
        nodes=(
            PipelineNode("a", "digits"),
            PipelineNode("b", "downstream"),
        ),
        edges=(PipelineEdge("e1", "a", "b", "mean_logits"),),
    )
    seen: List[str] = []

    results = run_pipeline(
        graph,
        request={"frames": [_raw_image()], "encoded": False},
        checkpoint_loader=_bundle_for,
        on_node_done=lambda node_id, _payload: seen.append(node_id),
    )

    assert seen == ["a", "b"]
    assert set(results) == {"a", "b"}
    assert "predicted" in results["a"]
    assert "predicted" in results["b"]


def test_a_single_source_node_uses_the_request_directly() -> None:
    """A lone node with no incoming edge reads the run request's frames."""
    _save_checkpoint("solo", num_classes=2)
    graph = PipelineGraph(
        name="g", nodes=(PipelineNode("a", "solo"),), edges=()
    )

    results = run_pipeline(
        graph,
        request={"frames": [_raw_image()], "encoded": False},
        checkpoint_loader=_bundle_for,
    )

    assert set(results) == {"a"}


def test_fan_in_is_refused() -> None:
    """A node with two incoming edges raises rather than picking one."""
    _save_checkpoint("a_ckpt", num_classes=2)
    _save_checkpoint("b_ckpt", num_classes=2)
    _save_checkpoint("c_ckpt", num_classes=2)
    graph = PipelineGraph(
        name="g",
        nodes=(
            PipelineNode("a", "a_ckpt"),
            PipelineNode("b", "b_ckpt"),
            PipelineNode("c", "c_ckpt"),
        ),
        edges=(
            PipelineEdge("e1", "a", "c", "mean_logits"),
            PipelineEdge("e2", "b", "c", "mean_logits"),
        ),
    )

    with pytest.raises(PipelineRunError, match="fan-in"):
        run_pipeline(
            graph,
            request={"frames": [_raw_image()], "encoded": False},
            checkpoint_loader=_bundle_for,
        )


def test_should_stop_halts_before_the_next_node() -> None:
    """A should_stop that's already true skips every node."""
    _save_checkpoint("solo", num_classes=2)
    graph = PipelineGraph(
        name="g", nodes=(PipelineNode("a", "solo"),), edges=()
    )

    results = run_pipeline(
        graph,
        request={"frames": [_raw_image()], "encoded": False},
        checkpoint_loader=_bundle_for,
        should_stop=lambda: True,
    )

    assert results == {}


def test_a_missing_checkpoint_raises_a_named_node_error() -> None:
    """A node referencing a checkpoint that doesn't exist names the node."""
    graph = PipelineGraph(
        name="g", nodes=(PipelineNode("a", "ghost"),), edges=()
    )

    with pytest.raises(PipelineRunError, match="node 'a'"):
        run_pipeline(
            graph,
            request={"frames": [_raw_image()], "encoded": False},
            checkpoint_loader=_bundle_for,
        )
