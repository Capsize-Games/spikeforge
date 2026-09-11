"""Topology-general inference contract built on the shared simulator."""

from typing import Any, Dict, List, Tuple

import pytest
import torch

from spikeforge.network import inference
from spikeforge.simulator import input_shape
from spikeforge.topology import registry

_Cases = Tuple[str, Dict[str, Any], str]

_CASES: List[_Cases] = [
    ("fc_legacy", {"hidden": 8, "beta": 0.5, "num_classes": 4}, "_lif1"),
    ("conv_net", {"channels": 2, "num_classes": 4}, "lif2"),
    ("recurrent_net", {"hidden": 6, "beta": 0.9, "num_classes": 4}, "lif2"),
]

_PAYLOAD_KEYS = {
    "predicted",
    "confidence",
    "true_label",
    "class_spikes",
    "output_over_time",
    "coding",
    "input_mode",
    "num_steps",
    "_rasters",
}


def _infer(name: str, params: Dict[str, Any]) -> Tuple[Any, Any]:
    """Build a topology and infer on a shaped fixed-seed spike train."""
    torch.manual_seed(0)
    spec, module = registry.build_topology(name, params)
    spikes = input_shape.to_input_shape(torch.rand(5, 1, 784), spec)
    classes = int(params["num_classes"])
    payload = inference.infer_spikes(module, spikes, classes)
    return spec, payload


@pytest.mark.parametrize("name,params,_hidden", _CASES)
def test_payload_keys_are_unchanged(
    name: str, params: Dict[str, Any], _hidden: str
) -> None:
    """Every topology emits exactly the historical payload keys."""
    _, payload = _infer(name, params)
    assert set(payload) == _PAYLOAD_KEYS
    assert payload["num_steps"] == 5
    assert payload["true_label"] is None
    assert len(payload["output_over_time"]) == 5
    assert len(payload["class_spikes"]) == 4


@pytest.mark.parametrize("name,params,hidden", _CASES)
def test_hidden_stage_is_located_per_topology(
    name: str, params: Dict[str, Any], hidden: str
) -> None:
    """The hidden raster comes from the neuron stage before the output."""
    spec, payload = _infer(name, params)
    assert inference.hidden_stage(spec) == hidden
    raster = payload["_rasters"]["output"]
    assert raster["num_steps"] == 5
    assert raster["num_neurons"] == 4


def test_conv_hidden_raster_is_bounded() -> None:
    """A spatial hidden stage flattens to a bounded neuron axis."""
    _, payload = _infer("conv_net", {"channels": 2, "num_classes": 4})
    hidden = payload["_rasters"]["hidden"]
    assert hidden["num_steps"] == 5
    assert hidden["num_neurons"] == 256


def test_membrane_key_is_additive() -> None:
    """``membrane=True`` adds traces without disturbing the old keys."""
    torch.manual_seed(0)
    spec, module = registry.build_topology(
        "recurrent_net", {"hidden": 6, "beta": 0.9, "num_classes": 4}
    )
    spikes = input_shape.to_input_shape(torch.rand(4, 1, 784), spec)
    base = inference.infer_spikes(module, spikes, 4)
    traced = inference.infer_spikes(module, spikes, 4, membrane=True)
    assert set(base) <= set(traced)
    assert "membrane" not in base
    assert set(traced["membrane"]) == {"lif1", "lif2", "out"}
