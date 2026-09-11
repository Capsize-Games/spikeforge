"""Per-stage heterogeneous neuron resolution and back-compatibility."""

from typing import Any, Dict

import pytest

from spikeforge.network.spiking_net import SpikingNet
from spikeforge.neurons.registry import NEURONS
from spikeforge.nir_bridge import to_nir
from spikeforge.topology import registry
from spikeforge.topology.spec import TopologySpec
from spikeforge.training.training_engine import TrainingEngine

pytest.importorskip("nir")

_SMALL: Dict[str, Any] = {"hidden": 4, "beta": 0.9, "num_classes": 3}
_LEGACY_KEYS = {
    "_fc1.weight", "_fc1.bias", "_fc2.weight", "_fc2.bias",
}


def _kinds(spec: TopologySpec) -> Dict[str, str]:
    """Return the kind of every neuron stage, keyed by name."""
    return {
        stage.name: stage.kind
        for stage in spec.stages
        if stage.kind in NEURONS
    }


def test_per_stage_override_changes_only_named_stage() -> None:
    """``neurons`` selects one stage's kind and leaves the rest default."""
    spec, _ = registry.build_topology(
        "fc_small", {**_SMALL, "neurons": {"lif1": "synaptic"}}
    )
    assert _kinds(spec) == {"lif1": "synaptic", "lif2": "leaky"}


def test_default_build_uses_leaky_everywhere() -> None:
    """Without overrides every neuron stage stays leaky."""
    spec, _ = registry.build_topology("fc_small", _SMALL)
    assert set(_kinds(spec).values()) == {"leaky"}


def test_stage_params_merge_over_computed_params() -> None:
    """``stage_params`` layers threshold/reset over the computed params."""
    spec, _ = registry.build_topology(
        "fc_small",
        {
            **_SMALL,
            "stage_params": {"lif1": {"threshold": 1.5, "reset": "zero"}},
        },
    )
    params = spec.stage("lif1").params
    assert params["threshold"] == 1.5
    assert params["reset"] == "zero"
    assert params["beta"] == 0.9


def test_heterogeneous_spec_round_trips() -> None:
    """A heterogeneous spec survives to_dict/from_dict unchanged."""
    spec, _ = registry.build_topology(
        "fc_small", {**_SMALL, "neurons": {"lif1": "synaptic"}}
    )
    assert TopologySpec.from_dict(spec.to_dict()) == spec


def test_fc_legacy_default_is_still_spiking_net() -> None:
    """The default legacy build keeps the wrapper and its state-dict keys."""
    spec, module = registry.build_topology(
        "fc_legacy", {"hidden": 4, "beta": 0.5, "num_classes": 3}
    )
    assert isinstance(module, SpikingNet)
    assert set(module.state_dict()) >= _LEGACY_KEYS
    assert set(_kinds(spec).values()) == {"leaky"}


def test_fc_legacy_override_uses_generic_module() -> None:
    """A per-stage override falls back to the generic stage module."""
    _spec, module = registry.build_topology(
        "fc_legacy",
        {
            "hidden": 4, "beta": 0.5, "num_classes": 3,
            "neurons": {"_lif1": "synaptic"},
        },
    )
    assert not isinstance(module, SpikingNet)
    assert type(module.get_submodule("_lif1")).__name__ == "Synaptic"


def test_engine_meta_reports_stage_neurons() -> None:
    """A checkpoint's meta carries a readable per-stage neuron summary."""
    engine = TrainingEngine(
        dataset="mnist", hidden=4, beta=0.9, num_steps=2, device="cpu",
        topology="fc_small",
        topology_params={**_SMALL, "neurons": {"lif1": "synaptic"}},
    )
    assert engine.spec.stage("lif1").kind == "synaptic"
    meta = engine._meta()
    assert meta["stage_neurons"] == {"lif1": "synaptic", "lif2": "leaky"}
    # The checkpoint's stored spec round-trips the heterogeneous layout.
    assert TopologySpec.from_dict(meta["spec"]) == engine.spec


def test_heterogeneous_stages_export_side_by_side() -> None:
    """A synaptic and a leaky stage export distinct NIR node kinds."""
    spec, module = registry.build_topology(
        "fc_small", {**_SMALL, "neurons": {"lif1": "synaptic"}}
    )
    graph = to_nir(spec, module)
    kinds = {type(node).__name__ for node in graph.nodes.values()}
    assert "CubaLIF" in kinds
    assert "LI" in kinds
