"""Tests for the Alpha neuron handler and the neuron comparison helper."""

import math

import pytest
import torch

from spikeforge.introspection.comparison import compare_neurons
from spikeforge.neurons import contract
from spikeforge.neurons.alpha import AlphaNeuron
from spikeforge.neurons.registry import build_neuron, neuron_kinds
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.topology.stage import Stage

pytest.importorskip("nir")


def test_alpha_is_registered() -> None:
    """The alpha kind is registered and builds an snnTorch module."""
    assert "alpha" in neuron_kinds()
    assert isinstance(build_neuron("alpha", {}), torch.nn.Module)


def test_alpha_runs_one_step_with_three_states() -> None:
    """A step returns spikes and the (syn_exc, syn_inh, mem) state."""
    handler = AlphaNeuron()
    module = handler.build({"alpha": 0.9, "beta": 0.8})
    spikes, state = handler.step(module, torch.rand(2, 4), None)
    assert spikes.shape == (2, 4)
    assert len(state) == 3
    assert all(tensor.shape == (2, 4) for tensor in state)


def test_alpha_exposes_canonical_params() -> None:
    """Alpha emits the base keys plus the tau_syn synaptic extra."""
    params = AlphaNeuron().nir_params({"alpha": 0.9, "beta": 0.8})
    assert set(params) == set(contract.BASE_KEYS) | set(
        contract.SYNAPTIC_KEYS
    )
    assert params["kind"] == "alpha"
    assert params["tau_syn"] == pytest.approx(-1.0 / math.log(0.9))
    assert params["tau_mem"] == pytest.approx(-1.0 / math.log(0.8))


def test_alpha_export_raises_typed_error() -> None:
    """The installed nir cannot map alpha, so export fails loudly."""
    with pytest.raises(UnsupportedStageError) as excinfo:
        map_stage(Stage("a", "alpha", {"alpha": 0.9, "beta": 0.8}))
    assert excinfo.value.kind == "alpha"
    assert "alpha" in str(excinfo.value)


def test_compare_neurons_runs_every_kind() -> None:
    """Every registered neuron kind runs on one shared spike train."""
    spikes = (torch.rand(4, 2, 3) > 0.5).float()
    result = compare_neurons(spikes)
    assert set(result) == set(neuron_kinds())
    for trajectory in result.values():
        assert "neuron" in trajectory.spikes
        assert "neuron" in trajectory.currents


def test_compare_neurons_shares_the_input_current() -> None:
    """The seeded linear front end feeds every kind the same current."""
    spikes = (torch.rand(4, 2, 3) > 0.5).float()
    result = compare_neurons(spikes)
    reference = result["leaky"].currents["neuron"]
    for trajectory in result.values():
        assert torch.allclose(trajectory.currents["neuron"], reference)
