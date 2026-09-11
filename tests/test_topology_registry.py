"""Tests for the topology registry lookup."""

import pytest

from snn_interpreter.topology import registry
from snn_interpreter.topology.spec import TopologySpec
from snn_interpreter.topology.stage_module import StageModule

_NAMES = (
    "fc_legacy", "fc_small", "conv_net", "recurrent_net",
    "sequence_mlp", "sequence_attn",
)


def test_topology_names_match_presets() -> None:
    """The registry lists exactly the shipped topologies."""
    assert set(registry.topology_names()) == set(_NAMES)


@pytest.mark.parametrize("name", _NAMES)
def test_build_topology_returns_valid_pair(name: str) -> None:
    """Every topology builds a validated spec and a runnable module."""
    spec, module = registry.build_topology(name)
    assert isinstance(spec, TopologySpec)
    assert isinstance(module, StageModule)
    assert spec.stage(spec.input).name == spec.input
    assert spec.stage(spec.output).name == spec.output


def test_params_override_defaults_and_drop_unknown() -> None:
    """Accepted params override defaults; unknown keys are ignored."""
    spec, _ = registry.build_topology(
        "fc_legacy", {"hidden": 5, "num_classes": 3, "mystery": 1}
    )
    assert spec.stage("_fc1").params["out_features"] == 5
    assert spec.stage("_fc2").params["out_features"] == 3


def test_resolved_params_copies_defaults() -> None:
    """``resolved_params`` returns a fresh mapping per call."""
    first = registry.resolved_params("conv_net")
    first["channels"] = 99
    assert registry.resolved_params("conv_net")["channels"] == 8


def test_unknown_topology_raises() -> None:
    """An unknown name is reported, never silently defaulted."""
    with pytest.raises(ValueError):
        registry.build_topology("nope")
