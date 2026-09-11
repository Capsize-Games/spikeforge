"""Topology ``neuron``/``surrogate`` parameter behaviour end to end."""

from typing import Any, Dict, List

import pytest
import torch

from spikeforge.introspection.surrogate import list_surrogates
from spikeforge.network.spiking_net import SpikingNet
from spikeforge.neurons.registry import NEURONS
from spikeforge.nir_bridge import to_nir
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.topology import registry
from spikeforge.training.training_engine import TrainingEngine

_SMALL: Dict[str, Any] = {"hidden": 4, "beta": 0.9, "num_classes": 3}
_LEGACY_KEYS = {"_fc1.weight", "_fc1.bias", "_fc2.weight", "_fc2.bias"}


def _engine(params: Dict[str, Any]) -> TrainingEngine:
    """Build a small download-free engine for a neuron-param topology."""
    return TrainingEngine(
        dataset="mnist", hidden=4, beta=0.9, num_steps=4, device="cpu",
        topology="fc_small", topology_params=params,
    )


def _neuron_stages(spec: Any) -> List[Any]:
    """Return the spec's neuron stages in declaration order."""
    return [stage for stage in spec.stages if stage.kind in NEURONS]


@pytest.mark.parametrize("neuron", ["synaptic", "lapicque"])
def test_neuron_param_trains_and_infers(neuron: str) -> None:
    """A non-default neuron kind builds, trains one step, and infers."""
    torch.manual_seed(0)
    engine = _engine({**_SMALL, "neuron": neuron})
    assert {stage.kind for stage in _neuron_stages(engine.spec)} == {neuron}
    metrics = engine._train_batch(
        torch.rand(4, 1, 28, 28), torch.randint(0, 3, (4,))
    )
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert engine.infer(torch.rand(4, 1, 784))["num_steps"] == 4


@pytest.mark.parametrize("neuron", ["synaptic", "lapicque"])
def test_neuron_param_exports_to_nir(neuron: str) -> None:
    """A mappable neuron kind exports to a non-empty NIR graph."""
    spec, module = registry.build_topology(
        "fc_small", {**_SMALL, "neuron": neuron}
    )
    assert to_nir(spec, module).nodes


def test_alpha_builds_but_nir_export_raises() -> None:
    """``alpha`` simulates but has no faithful NIR mapping."""
    spec, module = registry.build_topology("fc_legacy", {"neuron": "alpha"})
    assert isinstance(module, torch.nn.Module)
    assert not isinstance(module, SpikingNet)
    with pytest.raises(UnsupportedStageError):
        to_nir(spec, module)


def test_surrogate_threads_into_every_neuron_stage() -> None:
    """``surrogate`` reaches each neuron stage's params."""
    name = next(item for item in list_surrogates() if item != "LSO")
    spec, _ = registry.build_topology(
        "fc_small", {**_SMALL, "surrogate": name}
    )
    stages = _neuron_stages(spec)
    assert stages
    assert all(stage.params["surrogate"] == name for stage in stages)


def test_default_specs_omit_surrogate_and_use_leaky() -> None:
    """Without overrides neuron stages stay leaky and surrogate-free."""
    spec, _ = registry.build_topology("fc_small", _SMALL)
    stages = _neuron_stages(spec)
    assert {stage.kind for stage in stages} == {"leaky"}
    assert all("surrogate" not in stage.params for stage in stages)


def test_default_resolved_params_select_leaky() -> None:
    """Every preset resolves a leaky neuron and no surrogate by default."""
    for name in registry.topology_names():
        resolved = registry.resolved_params(name)
        assert resolved["neuron"] == "leaky"
        assert resolved["surrogate"] is None


def test_fc_legacy_default_keeps_legacy_wrapper() -> None:
    """fc_legacy's default still renders the byte-identical SpikingNet."""
    spec, module = registry.build_topology(
        "fc_legacy", {"hidden": 4, "beta": 0.5, "num_classes": 3}
    )
    assert isinstance(module, SpikingNet)
    assert set(module.state_dict()) >= _LEGACY_KEYS
    assert {stage.kind for stage in _neuron_stages(spec)} == {"leaky"}
