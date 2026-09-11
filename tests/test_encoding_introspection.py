"""Tests for encoding introspection and per-coding image reconstruction."""

import json

import pytest
import torch

from spikeforge.encoding.spike_encoder import SpikeEncoder
from spikeforge.introspection.decoding import decode_delta
from spikeforge.introspection.encoding import encoding_report


def _rate_image() -> torch.Tensor:
    """Return a fixed ``1x8x8`` image for deterministic reconstruction."""
    torch.manual_seed(1234)
    return torch.rand(1, 8, 8)


def _correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    """Return the Pearson correlation of two flattened tensors."""
    a = left.reshape(-1).float()
    b = right.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


def test_rate_reconstruction_tracks_the_input_and_has_its_shape() -> None:
    """A seeded rate reconstruction correlates with the image, same shape."""
    image = _rate_image()
    torch.manual_seed(7)
    report = encoding_report(
        image, SpikeEncoder(coding="rate", num_steps=200)
    )
    assert report["reconstruction_supported"] is True
    reconstruction = torch.tensor(report["reconstruction"])
    assert list(reconstruction.shape) == [1, 8, 8]
    assert _correlation(reconstruction, image) > 0.8


def test_rate_report_is_json_serialisable() -> None:
    """The whole report survives ``json.dumps`` with no tensor conversion."""
    torch.manual_seed(7)
    report = encoding_report(
        _rate_image(), SpikeEncoder(coding="rate", num_steps=20)
    )
    assert isinstance(report["firing_rate"], float)
    assert isinstance(report["sparsity"], float)
    assert isinstance(json.dumps(report), str)


def test_latency_decode_is_a_monotonic_inverse() -> None:
    """Higher intensity fires earlier, so it decodes to a higher value.

    Approximation: the spike train quantises time-to-first-spike to integer
    steps (and normalises it to ``num_steps``), so the decoded value matches
    the intensity only within ``1 / (num_steps - 1)``.
    """
    image = torch.tensor([[[0.9, 0.2, 0.05]]])
    report = encoding_report(
        image, SpikeEncoder(coding="latency", num_steps=100)
    )
    decoded = report["reconstruction"][0][0]
    assert decoded[0] > decoded[1] > decoded[2]
    assert decoded[0] == pytest.approx(0.9, abs=0.02)
    assert decoded[1] == pytest.approx(0.2, abs=0.02)
    assert report["stats"]["linear"] is True


def test_delta_decode_integrates_the_on_off_stream() -> None:
    """Integration credits one delta threshold of change per spike."""
    spikes = torch.zeros(5, 1, 1)
    spikes[1, 0, 0] = 1.0
    spikes[3, 0, 0] = 1.0
    decoded = decode_delta(spikes, threshold=0.5)
    assert float(decoded[0]) == pytest.approx(1.0)
    integrated = spikes.cumsum(dim=0).reshape(-1)
    assert bool((integrated[1:] >= integrated[:-1]).all())


def test_delta_report_is_a_documented_lower_bound_on_a_ramp() -> None:
    """Integrating a rising ramp's stream is a lower-bound pixel estimate.

    Approximation: ``spikegen.delta`` emits one spike per step rising by at
    least the threshold, whatever the size of the rise, so the integral
    credits exactly one threshold per spike and never exceeds the pixel.
    """
    image = torch.full((1, 1, 1), 0.6)
    encoder = SpikeEncoder(coding="delta", num_steps=11)
    spikes = encoder.encode_image(image)
    report = encoding_report(image, encoder)
    scale = encoder.delta_threshold / 100.0
    decoded = report["reconstruction"][0][0][0]
    assert decoded == pytest.approx(float(spikes.sum()) * scale)
    assert 0.0 < decoded <= 0.6


def test_random_coding_reports_no_reconstruction_signal() -> None:
    """The random coding carries no image signal, so reconstruction is None."""
    report = encoding_report(
        _rate_image(), SpikeEncoder(coding="random", num_steps=20, seed=5)
    )
    assert report["reconstruction"] is None
    assert report["reconstruction_supported"] is False
    assert report["stats"]["image_signal"] is False
    assert "no image signal" in report["approximation"]
    assert isinstance(json.dumps(report), str)
