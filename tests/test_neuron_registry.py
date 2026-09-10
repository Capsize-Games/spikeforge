"""Tests for the spiking neuron registry and NIR parameter contract."""

from math import log

import pytest
import torch

from snn_interpreter.neurons import contract
from snn_interpreter.neurons.registry import (
    NEURONS,
    build_neuron,
    neuron_kinds,
)

_KINDS = ("leaky", "lapicque", "synaptic", "recurrent")
_PARAMS = {
    "leaky": {},
    "lapicque": {},
    "synaptic": {"alpha": 0.8, "beta": 0.9},
    "recurrent": {"beta": 0.9, "linear_features": 4},
}


def test_registry_exposes_documented_kinds() -> None:
    """Every documented neuron kind is registered exactly once."""
    assert set(neuron_kinds()) == set(_KINDS)
    assert set(NEURONS) == set(_KINDS)


@pytest.mark.parametrize("kind", _KINDS)
def test_every_kind_builds(kind: str) -> None:
    """Each handler builds its snnTorch module without error."""
    assert isinstance(build_neuron(kind, _PARAMS[kind]), torch.nn.Module)


def test_unknown_kind_raises() -> None:
    """An unregistered neuron kind is rejected."""
    with pytest.raises(KeyError):
        build_neuron("not-a-neuron", {})


@pytest.mark.parametrize("kind", _KINDS)
def test_base_contract_keys_are_present(kind: str) -> None:
    """All kinds emit the shared NIR base keys and constants."""
    params = NEURONS[kind].nir_params(_PARAMS[kind])
    assert set(contract.BASE_KEYS) <= set(params)
    assert params["kind"] == kind
    assert params["dt"] == 1.0
    assert params["v_leak"] == 0.0
    decay = contract.decay_from_tau(params["tau_mem"])
    assert params["R"] == pytest.approx(1.0 / (1.0 - decay))


@pytest.mark.parametrize(
    ("kind", "extras"),
    [
        ("leaky", ()),
        ("lapicque", ()),
        ("synaptic", contract.SYNAPTIC_KEYS),
        ("recurrent", contract.RECURRENT_KEYS),
    ],
)
def test_nir_param_key_sets_are_exact(
    kind: str, extras: tuple
) -> None:
    """Each kind emits exactly the base keys plus its documented extras."""
    params = NEURONS[kind].nir_params(_PARAMS[kind])
    assert set(params) == set(contract.BASE_KEYS) | set(extras)


@pytest.mark.parametrize("beta", [0.5, 0.9, 0.99])
def test_tau_mem_matches_beta_relation(beta: float) -> None:
    """tau_mem is consistent with beta under beta = exp(-dt / tau)."""
    params = NEURONS["leaky"].nir_params({"beta": beta})
    assert params["tau_mem"] == pytest.approx(-1.0 / log(beta))
    restored = contract.decay_from_tau(params["tau_mem"])
    assert restored == pytest.approx(beta)


def test_reset_mapping_subtract_and_zero() -> None:
    """Subtract maps to threshold - 1 and zero maps to 0."""
    subtract = NEURONS["leaky"].nir_params({"threshold": 1.5})
    zero = NEURONS["leaky"].nir_params(
        {"threshold": 1.5, "reset": "zero"}
    )
    assert subtract["v_threshold"] == 1.5
    assert subtract["v_reset"] == pytest.approx(0.5)
    assert zero["v_reset"] == 0.0


def test_unknown_reset_is_rejected() -> None:
    """An unknown reset mechanism raises rather than guessing."""
    with pytest.raises(ValueError):
        NEURONS["leaky"].nir_params({"reset": "bogus"})


def test_synaptic_emits_two_time_constants() -> None:
    """Synaptic neurons expose both synaptic and membrane constants."""
    params = NEURONS["synaptic"].nir_params(
        {"alpha": 0.8, "beta": 0.9}
    )
    assert params["tau_syn"] == pytest.approx(-1.0 / log(0.8))
    assert params["tau_mem"] == pytest.approx(-1.0 / log(0.9))


def test_recurrent_declares_feedback_delay() -> None:
    """Recurrent neurons expose the one-step feedback delay flag."""
    assert NEURONS["recurrent"].nir_params(
        {"linear_features": 4}
    )["feedback_delay"] is True


@pytest.mark.parametrize("kind", _KINDS)
def test_initial_state_is_empty_on_device(kind: str) -> None:
    """Initial neuron state is zero-length and lives on the given device."""
    state = NEURONS[kind].initial_state(torch.device("cpu"))
    assert state
    assert all(tensor.numel() == 0 for tensor in state)
    assert all(tensor.device.type == "cpu" for tensor in state)
