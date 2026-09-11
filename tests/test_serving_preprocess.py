"""The single pure ``encode`` call site shared by train and serve."""

from typing import Any

import pytest
import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.serving import preprocess
from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.serving.errors import EncodeSpecError
from spikeforge.topology import registry


def _images(batch: int = 2, seed: int = 0) -> torch.Tensor:
    """Return a deterministic ``[B, 1, 28, 28]`` image batch."""
    torch.manual_seed(seed)
    return torch.rand(batch, 1, 28, 28)


def test_image_batch_becomes_flat_spikes() -> None:
    """A 4-D batch encodes to the flat ``[T, B, F]`` code."""
    spec = EncodeSpec(coding="latency", num_steps=5)
    spikes = preprocess.encode(_images(), spec)
    assert spikes.shape == (5, 2, 28 * 28)


def test_single_image_is_promoted_to_a_batch() -> None:
    """A 3-D image is promoted to a single-item batch."""
    spec = EncodeSpec(coding="latency", num_steps=4)
    spikes = preprocess.encode(torch.rand(1, 28, 28), spec)
    assert spikes.shape == (4, 1, 28 * 28)


def test_matches_the_spike_encoder_directly() -> None:
    """The shared call site is exactly ``SpikeEncoder.encode``."""
    images = _images()
    spec = EncodeSpec(coding="latency", num_steps=5)
    expected = SpikeEncoder(coding="latency", num_steps=5).encode(images)
    assert torch.equal(preprocess.encode(images, spec), expected)


def test_latency_coding_is_deterministic() -> None:
    """A latency code has no RNG, so it repeats exactly."""
    images = _images()
    spec = EncodeSpec(coding="latency", num_steps=5)
    assert torch.equal(
        preprocess.encode(images, spec), preprocess.encode(images, spec)
    )


def test_a_pinned_seed_makes_a_random_coding_repeat() -> None:
    """``random_seed`` reseeds before dispatch, so the code is repeatable."""
    images = _images()
    spec = EncodeSpec(coding="rate", num_steps=4, random_seed=2024)
    assert torch.equal(
        preprocess.encode(images, spec), preprocess.encode(images, spec)
    )


def test_conv_topology_reshapes_spatially() -> None:
    """A conv input receives ``[T, B, C, H, W]`` when a geometry is given."""
    spec, _ = registry.build_topology(
        "conv_net", {"channels": 2, "num_classes": 4}
    )
    spikes = preprocess.encode(
        _images(3),
        EncodeSpec(coding="latency", num_steps=4),
        spec,
        geometry=(28, 28),
    )
    assert spikes.shape == (4, 3, 1, 28, 28)


def test_flat_topology_stays_flat() -> None:
    """A feature input keeps its flat ``[T, B, F]`` layout."""
    spec, _ = registry.build_topology(
        "recurrent_net", {"hidden": 9, "num_classes": 4}
    )
    spikes = preprocess.encode(
        _images(3), EncodeSpec(coding="latency", num_steps=4), spec
    )
    assert spikes.shape == (4, 3, 28 * 28)


def test_rejects_a_non_numeric_sample() -> None:
    """A payload the encoder cannot read is refused with a ``TypeError``."""
    with pytest.raises(TypeError):
        preprocess.encode("not an image", EncodeSpec(num_steps=4))


def test_rejects_a_bad_spec() -> None:
    """An invalid spec is refused before any encoding work happens."""
    with pytest.raises(EncodeSpecError):
        preprocess.encode(_images(), {"num_steps": 0})


def test_encode_batch_matches_encode() -> None:
    """``encode_batch`` is the same pure call as ``encode``."""
    images = _images(3)
    spec = EncodeSpec(coding="latency", num_steps=4)
    assert torch.equal(
        preprocess.encode_batch(images, spec), preprocess.encode(images, spec)
    )


def test_encoder_for_returns_a_configured_encoder() -> None:
    """``encoder_for`` validates the spec and builds the core encoder."""
    encoder = preprocess.encoder_for(EncodeSpec(coding="delta", num_steps=6))
    assert encoder.coding == "delta"
    assert encoder.num_steps == 6


def test_encoder_for_rejects_a_bad_spec() -> None:
    """``encoder_for`` refuses an unsupported coding."""
    bad: Any = {"coding": "bogus"}
    with pytest.raises(EncodeSpecError):
        preprocess.encoder_for(bad)
