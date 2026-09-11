"""Non-square sensor geometry: presets, transform, and input reshape."""

from typing import Any, Dict

import pytest
import torch

from server.schemas import EncodeConfig
from snn_interpreter.data import sample_source
from snn_interpreter.data.datasets import transform
from snn_interpreter.data.event_geometry import EventGeometryError
from snn_interpreter.data.image_size import as_size
from snn_interpreter.data.sample_source import SampleSource
from snn_interpreter.events import event_source
from snn_interpreter.simulator import input_shape
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module
from snn_interpreter.training.event_engine import EventTrainingEngine

_SIZE = (32, 28)


def _spatial_frames(steps: int = 4, batch: int = 2) -> torch.Tensor:
    """Return binary frames whose flat length matches ``_SIZE``."""
    flat = torch.rand(steps, batch, _SIZE[0] * _SIZE[1])
    return (flat > 0.5).to(torch.float32)


def test_conv_net_derives_features_from_a_non_square_shape() -> None:
    """conv_net pools a (32, 28) sensor to 8x7 features on the fc stage."""
    spec = presets.conv_net(
        in_channels=1, channels=2, num_classes=3, input_size=_SIZE
    )
    assert spec.stage("fc").params["in_features"] == 2 * 2 * 8 * 7


def test_non_square_input_builds_and_runs_through_conv_net() -> None:
    """A (32, 28) spike volume runs through conv_net end to end."""
    spec = presets.conv_net(
        in_channels=1, channels=2, num_classes=3, input_size=_SIZE
    )
    module = build_module(spec)
    spikes = input_shape.to_input_shape(_spatial_frames(), spec, _SIZE)
    assert list(spikes.shape) == [4, 2, 1, 32, 28]
    assert tuple(run(module, spikes).logits.shape) == (2, 3)


def test_to_input_shape_propagates_a_tuple() -> None:
    """The reshape helper accepts an explicit (H, W) pair."""
    spec = presets.conv_net(channels=2, num_classes=3, input_size=_SIZE)
    flat = torch.rand(3, 2, 32 * 28)
    shaped = input_shape.to_input_shape(flat, spec, _SIZE)
    assert list(shaped.shape) == [3, 2, 1, 32, 28]


def test_feature_topology_accepts_a_non_square_area() -> None:
    """fc_legacy with input_size = 32*28 accepts the flattened sensor."""
    spec = presets.fc_legacy(
        hidden=4, beta=0.5, num_classes=3, input_size=32 * 28
    )
    module = build_module(spec)
    flat = torch.rand(4, 2, 32 * 28)
    assert input_shape.to_input_shape(flat, spec) is flat
    assert tuple(run(module, flat).logits.shape) == (2, 3)


def test_conv_net_rejects_a_side_below_the_pool_window() -> None:
    """A side the two 2x2 pools cannot downsample is refused by name."""
    with pytest.raises(ValueError) as ctx:
        presets.conv_net(input_size=(3, 28))
    assert "3x28" in str(ctx.value)


def test_event_engine_accepts_a_declared_non_square_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conv topology built for (32, 28) accepts that event sensor."""
    monkeypatch.setattr(event_source, "SYNTHETIC_SHAPE", _SIZE)
    engine = EventTrainingEngine(
        dataset="n_mnist", synthetic_only=True, num_steps=4, device="cpu",
        topology="conv_net",
        topology_params={"input_size": _SIZE, "channels": 2},
    )
    inputs, targets = engine._epoch_batches()[0]
    assert list(inputs.shape)[-2:] == [32, 28]
    assert engine._train_batch(inputs, targets)["loss"] > 0.0


def test_event_engine_rejects_a_mismatched_non_square_sensor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A conv topology built for 28x28 refuses a 32x28 event sensor."""
    monkeypatch.setattr(event_source, "SYNTHETIC_SHAPE", _SIZE)
    with pytest.raises(EventGeometryError) as ctx:
        EventTrainingEngine(
            dataset="n_mnist", synthetic_only=True, num_steps=4,
            device="cpu", topology="conv_net",
            topology_params={"input_size": 28},
        )
    message = str(ctx.value)
    assert "28x28" in message
    assert "32x28" in message


def test_transform_resizes_to_the_requested_shape() -> None:
    """The transform resizes to ``size`` and defaults to 28x28."""
    assert transform().transforms[1].size == (28, 28)
    assert transform(_SIZE).transforms[1].size == _SIZE


def test_sample_source_passes_its_size_to_the_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The size accessor returns and forwards the requested geometry."""
    captured: Dict[str, Any] = {}

    def _fake_build(
        name: str, train: bool = True, download: bool = True,
        size: Any = (28, 28),
    ) -> Any:
        captured["size"] = size
        return []

    monkeypatch.setattr(sample_source, "build_dataset", _fake_build)
    source = SampleSource("mnist", size=_SIZE)
    assert captured["size"] == _SIZE
    assert source.size == _SIZE
    assert SampleSource("mnist").size == (28, 28)


def test_as_size_normalises_ints_and_pairs() -> None:
    """A square side and an explicit pair normalise to one convention."""
    assert as_size(28) == (28, 28)
    assert as_size((28, 28)) == (28, 28)
    assert as_size([32, 28]) == (32, 28)


def test_encode_config_input_size_is_additive() -> None:
    """EncodeConfig gains an optional geometry without changing defaults."""
    assert EncodeConfig().input_size is None
    assert EncodeConfig(input_size=[32, 28]).input_size == (32, 28)
