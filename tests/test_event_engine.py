"""Event engine shapes, polarity-aware rasters, and simulator drive."""

import json
from typing import Any, Dict, Optional, Tuple

from server.event_engine import EventEngine
from server.schemas import EncodeConfig
from snn_interpreter.events.event_source import EventSampleSource
from snn_interpreter.simulator import input_shape
from snn_interpreter.simulator.runner import run
from snn_interpreter.topology import registry
from snn_interpreter.topology.spec import TopologySpec

_STEPS = 10
_SIDE = 28
_FLAT = _SIDE * _SIDE


def _engine(spec: Optional[TopologySpec] = None) -> EventEngine:
    """Build an offline event engine, optionally shaped for a topology."""
    cfg = EncodeConfig(dataset="n_mnist")
    source = EventSampleSource("n_mnist", synthetic_only=True)
    return EventEngine(cfg, spec=spec, source=source)


def _topology(name: str, params: Dict[str, Any]) -> Tuple[TopologySpec, Any]:
    """Return a small deterministic ``(spec, module)`` pair."""
    return registry.build_topology(name, params)


def test_engine_reports_modality_and_sample_metadata() -> None:
    """The engine exposes the event modality and the sample's label."""
    engine = _engine()
    assert engine.modality == "event"
    assert engine.dataset == "n_mnist"
    assert engine.num_steps() == _STEPS
    assert engine.sample_index() == 0
    assert engine.target_label() == engine.sample_label()


def test_spike_input_uses_the_flat_feature_contract() -> None:
    """Without a spec the engine returns the shared ``[T,1,H*W]`` layout."""
    assert list(_engine().spike_input().shape) == [_STEPS, 1, _FLAT]


def test_spike_frame_is_a_polarity_stacked_grid() -> None:
    """The frame stacks the ON channel above the OFF channel."""
    engine = _engine()
    frame = engine.spike_frame(0)
    assert len(frame) == 2 * _SIDE
    assert len(frame[0]) == _SIDE
    assert len(engine.sample_frame()) == 2 * _SIDE


def test_raster_splits_on_and_off_neurons() -> None:
    """ON events occupy the first block, OFF the second."""
    raster = _engine().raster()
    assert raster["num_steps"] == _STEPS
    assert raster["num_neurons"] == 2 * _FLAT
    assert len(raster["time"]) == len(raster["neurons"])
    assert min(raster["neurons"]) < _FLAT
    assert max(raster["neurons"]) >= _FLAT
    assert json.dumps(raster)


def test_engine_has_no_image_or_reconstruction() -> None:
    """Event samples carry no static image and no rate reconstruction."""
    engine = _engine()
    assert engine.sample_image() is None
    assert engine.reconstruction() is None


def test_spec_shaped_input_drives_fc_small() -> None:
    """The shaped spikes run through the fc_small simulator loop."""
    spec, module = _topology(
        "fc_small", {"hidden": 4, "num_classes": 3}
    )
    result = run(module, _engine(spec).spike_input())
    assert list(result.logits.shape) == [1, 3]


def test_spec_shaped_input_drives_conv_net() -> None:
    """The shaped spikes run through the conv_net simulator loop."""
    spec, module = _topology(
        "conv_net", {"channels": 2, "num_classes": 3}
    )
    result = run(module, _engine(spec).spike_input())
    assert list(result.logits.shape) == [1, 3]


def test_flat_input_reshapes_for_a_spatial_topology() -> None:
    """The flat engine output reshapes through the shared helper."""
    spec, module = _topology(
        "conv_net", {"channels": 2, "num_classes": 3}
    )
    shaped = input_shape.to_input_shape(_engine().spike_input(), spec)
    assert list(shaped.shape) == [_STEPS, 1, 1, _SIDE, _SIDE]
    assert list(run(module, shaped).logits.shape) == [1, 3]
