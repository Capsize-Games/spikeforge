"""Event-to-spike bridge layout, simulator drive, and JSON metadata."""

import json

import pytest

from snn_interpreter.events.event_bridge import EventSpikeBridge
from snn_interpreter.events.event_sample import EventSample
from snn_interpreter.events.synthetic import moving_dot
from snn_interpreter.nir_bridge.validator import validate
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module


def _sample() -> EventSample:
    """Return a deterministic 8x8 stream with four events."""
    return moving_dot(num_steps=4, shape=(8, 8), start=(1.0, 1.0))


def test_feature_layout_is_flat() -> None:
    """A feature input stage receives ``[T, B, F]``."""
    spec = presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=64)
    spikes, meta = EventSpikeBridge().encode(_sample(), spec)
    assert list(spikes.shape) == [4, 1, 64]
    assert meta["spatial"] is False
    assert meta["bins"] == 4
    assert meta["num_events"] == 4


def test_spatial_layout_is_channel_height_width() -> None:
    """A spatial input stage receives ``[T, B, C, H, W]``."""
    spec = presets.conv_net(channels=2, num_classes=3, input_size=8)
    spikes, meta = EventSpikeBridge().encode(_sample(), spec)
    assert list(spikes.shape) == [4, 1, 1, 8, 8]
    assert meta["spatial"] is True
    assert meta["channels"] == 1
    assert meta["output_shape"] == [4, 1, 1, 8, 8]


def test_two_channel_spatial_layout_keeps_polarity() -> None:
    """A two-channel stage keeps the ON/OFF split."""
    spec = presets.conv_net(
        in_channels=2, channels=2, num_classes=3, input_size=8
    )
    spikes, meta = EventSpikeBridge().encode(_sample(), spec)
    assert list(spikes.shape) == [4, 1, 2, 8, 8]
    assert meta["channels"] == 2


def test_metadata_is_json_serialisable() -> None:
    """Conversion metadata is plain and ``json.dumps``-able."""
    spec = presets.conv_net(channels=2, num_classes=3, input_size=8)
    _, meta = EventSpikeBridge().encode(_sample(), spec)
    assert isinstance(meta["sparsity"], float)
    assert isinstance(json.dumps(meta), str)


def test_count_mode_accumulates_more_than_binary() -> None:
    """The count voxel never holds less than the binary frames."""
    spec = presets.fc_legacy(hidden=4, beta=0.5, num_classes=3, input_size=64)
    sample = _sample()
    binary, _ = EventSpikeBridge("binary").encode(sample, spec)
    counts, _ = EventSpikeBridge("count").encode(sample, spec)
    assert float(counts.sum()) >= float(binary.sum())


def test_bridge_output_drives_fc_small() -> None:
    """fc_small runs on the bridge's spatial tensor without error."""
    spec = presets.fc_small(hidden=4, beta=0.9, num_classes=3, input_size=64)
    module = build_module(spec)
    spikes, _ = EventSpikeBridge().encode(_sample(), spec)
    result = run(module, spikes)
    assert list(result.logits.shape) == [1, 3]


def test_bridge_output_drives_conv_net() -> None:
    """conv_net runs on the bridge's spatial tensor without error."""
    spec = presets.conv_net(channels=2, num_classes=3, input_size=8)
    module = build_module(spec)
    spikes, _ = EventSpikeBridge().encode(_sample(), spec)
    result = run(module, spikes)
    assert list(result.logits.shape) == [1, 3]


def test_bridge_output_validates_against_nir() -> None:
    """The bridge output is accepted by the independent NIR validator."""
    pytest.importorskip("nir")
    spec = presets.conv_net(channels=2, num_classes=3, input_size=8)
    module = build_module(spec)
    spikes, _ = EventSpikeBridge().encode(_sample(), spec)
    report = validate(spec, module, spikes)
    assert report["within_tolerance"] is True
    assert json.dumps(report)
