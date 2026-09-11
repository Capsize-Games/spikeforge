"""Neuron-state initialisation and extraction for the simulator."""

from typing import Any, Dict

import torch

from spikeforge.neurons.contract import NeuronState
from spikeforge.neurons.registry import NEURONS
from spikeforge.topology.spec import TopologySpec


def neuron_kinds(spec: TopologySpec) -> Dict[str, str]:
    """Map every neuron stage name to its registered kind."""
    return {
        stage.name: stage.kind
        for stage in spec.stages
        if stage.kind in NEURONS
    }


def initial_state(spec: TopologySpec, x: torch.Tensor) -> Dict[str, Any]:
    """Build a fresh per-stage neuron state placed on ``x``'s device.

    The state is keyed by stage name and holds the zero-length tuple each
    handler expects, so a neuron materialises its real tensors on the first
    step regardless of the ``(mem,)`` or ``(syn, mem)`` layout.
    """
    kinds = neuron_kinds(spec)
    return {
        name: NEURONS[kind].initial_state(x.device)
        for name, kind in kinds.items()
    }


def membrane_of(state: NeuronState) -> torch.Tensor:
    """Return the membrane tensor, which is last in every state layout."""
    return state[-1]
