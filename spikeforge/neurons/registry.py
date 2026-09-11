"""Registry mapping neuron kind strings to their handlers."""

from typing import Any, Dict, Mapping, Tuple

import torch.nn as nn

from spikeforge.neurons.alpha import AlphaNeuron
from spikeforge.neurons.handler import NeuronHandler
from spikeforge.neurons.lapicque import LapicqueNeuron
from spikeforge.neurons.leaky import LeakyNeuron
from spikeforge.neurons.recurrent import RecurrentNeuron
from spikeforge.neurons.synaptic import SynapticNeuron

#: Stateless handlers keyed by kind; one instance is shared across callers.
NEURONS: Dict[str, NeuronHandler] = {
    LeakyNeuron.kind: LeakyNeuron(),
    LapicqueNeuron.kind: LapicqueNeuron(),
    SynapticNeuron.kind: SynapticNeuron(),
    RecurrentNeuron.kind: RecurrentNeuron(),
    AlphaNeuron.kind: AlphaNeuron(),
}


def neuron_kinds() -> Tuple[str, ...]:
    """Return the registered neuron kinds in registration order."""
    return tuple(NEURONS)


def handler(kind: str) -> NeuronHandler:
    """Return the handler for ``kind``; raise ``KeyError`` when unknown."""
    if kind not in NEURONS:
        raise KeyError(f"unknown neuron kind: {kind!r}")
    return NEURONS[kind]


def build_neuron(kind: str, params: Mapping[str, Any]) -> nn.Module:
    """Build the snnTorch module for ``kind`` from ``params``."""
    return handler(kind).build(params)
