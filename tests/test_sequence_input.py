"""Sequence-layout frames flow through the simulator unchanged."""

import torch

from snn_interpreter.data import sequence_source
from snn_interpreter.data.datasets import dataset_modality
from snn_interpreter.simulator import input_shape
from snn_interpreter.simulator.frames import normalise_frame
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import registry


def test_sequence_frames_keep_their_layout() -> None:
    """A ``[T, B, L, D]`` sequence keeps its token axis through the run."""
    spec, module = registry.build_topology("sequence_mlp")
    frames = sequence_source.random_frames(5, 2, 8, 8, 0)
    trajectory = run(module, frames, track=True)
    assert tuple(trajectory.logits.shape) == (2, 8, 4)


def test_normalise_frame_passes_sequence_frames() -> None:
    """Multi-dimensional frames pass through; flat feature frames collapse."""
    frame = torch.rand(2, 8, 8)
    assert normalise_frame(frame, "linear") is frame
    flat = torch.rand(2, 8)
    assert normalise_frame(flat, "linear").shape == (2, 8)


def test_to_input_shape_leaves_sequence_frames() -> None:
    """A sequence topology keeps its caller-supplied layout unchanged."""
    spec, _ = registry.build_topology("sequence_mlp")
    frames = torch.rand(4, 2, 8, 8)
    assert input_shape.to_input_shape(frames, spec) is frames


def test_to_input_shape_still_reshapes_spatial() -> None:
    """Spatial topologies still receive ``[T, B, C, H, W]`` frames."""
    spec, _ = registry.build_topology("conv_net", {"input_size": 28})
    flat = torch.rand(4, 2, 28 * 28)
    assert input_shape.to_input_shape(flat, spec).shape == (4, 2, 1, 28, 28)


def test_sequence_source_serves_tokens_and_label() -> None:
    """The toy source returns in-vocabulary tokens and a parity label."""
    source = sequence_source.SequenceSource(length=6, vocab=10, seed=1)
    tokens, label = source.sample()
    assert tuple(tokens.shape) == (6,)
    assert int(tokens.min()) >= 0
    assert int(tokens.max()) < 10
    assert label in (0, 1)


def test_sequence_modality_is_reported() -> None:
    """The dataset registry advertises the new sequence modality."""
    assert dataset_modality("sequence_toy") == "sequence"
