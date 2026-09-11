"""Tests for selectable surrogate gradients and neuron surrogate wiring."""

import json
from typing import Any, Dict

import pytest
import torch
from snntorch import surrogate as surrogate_module

from spikeforge.introspection.surrogate import (
    list_surrogates,
    resolve_surrogate,
    surrogate_curve,
)
from spikeforge.neurons.registry import NEURONS, build_neuron
from spikeforge.neurons.spike_grad import spike_grad_from

_KINDS = ("leaky", "lapicque", "synaptic", "recurrent", "alpha")
_PARAMS: Dict[str, Dict[str, Any]] = {
    "leaky": {},
    "lapicque": {},
    "synaptic": {"alpha": 0.8, "beta": 0.9},
    "recurrent": {"beta": 0.9, "linear_features": 4},
    "alpha": {"alpha": 0.9, "beta": 0.8},
}
_CORE = (
    "atan",
    "fast_sigmoid",
    "sigmoid",
    "triangular",
    "heaviside",
    "SFS",
    "SSO",
    "spike_rate_escape",
    "straight_through_estimator",
)


def test_list_surrogates_matches_the_installed_module() -> None:
    """Every listed surrogate is exposed by ``snntorch.surrogate``."""
    names = list_surrogates()
    assert names
    exposed = {
        name for name in vars(surrogate_module) if not name.startswith("_")
    }
    assert set(names) <= exposed
    assert set(_CORE) <= set(names)
    assert "custom_surrogate" not in names


def test_resolve_surrogate_returns_a_callable() -> None:
    """A known name resolves to a callable; an unknown name is rejected."""
    assert callable(resolve_surrogate("atan"))
    with pytest.raises(KeyError):
        resolve_surrogate("not-a-surrogate")


def test_surrogate_curve_is_json_serialisable_with_a_gradient() -> None:
    """The derivative curve has equal-length axes and a nonzero response."""
    curve = surrogate_curve("atan", -1.0, 1.0, points=11)
    assert curve["name"] == "atan"
    assert len(curve["x"]) == 11
    assert len(curve["y"]) == 11
    assert max(abs(value) for value in curve["y"]) > 0.0
    assert isinstance(json.dumps(curve), str)


def test_surrogate_curve_rejects_too_few_points() -> None:
    """Fewer than two sample points cannot define a curve."""
    with pytest.raises(ValueError):
        surrogate_curve("atan", 0.0, 1.0, points=1)


def test_selected_surrogate_builds_and_runs_one_step() -> None:
    """A neuron built with a chosen surrogate runs a single step."""
    module = build_neuron("leaky", {"surrogate": "fast_sigmoid"})
    spikes, membrane = module(torch.ones(4), torch.zeros(4))
    assert spikes.shape == torch.Size([4])
    assert membrane.shape == torch.Size([4])


def test_unknown_surrogate_is_rejected_at_build() -> None:
    """A misspelled surrogate name fails loudly rather than falling back."""
    with pytest.raises(KeyError):
        build_neuron("leaky", {"surrogate": "not-a-surrogate"})


def test_absent_surrogate_resolves_to_none() -> None:
    """The default build parameter is ``None``, preserving snnTorch's path."""
    assert spike_grad_from({}) is None
    assert spike_grad_from({"surrogate": None}) is None


def _step_output(kind: str, params: Dict[str, Any]) -> torch.Tensor:
    """Build ``kind`` under a fixed seed and return one step of spikes."""
    torch.manual_seed(0)
    module = build_neuron(kind, params)
    spikes, _ = NEURONS[kind].step(module, torch.ones(4), None)
    return spikes


@pytest.mark.parametrize("kind", _KINDS)
def test_default_surrogate_is_byte_identical(kind: str) -> None:
    """Omitting ``surrogate`` matches passing ``None`` exactly."""
    params = dict(_PARAMS[kind])
    without = _step_output(kind, params)
    with_none = _step_output(kind, {**params, "surrogate": None})
    assert torch.equal(without, with_none)
    assert callable(build_neuron(kind, params).spike_grad)
