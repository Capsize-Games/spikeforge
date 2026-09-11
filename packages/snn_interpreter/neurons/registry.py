"""Registry mapping neuron kind strings to their handlers."""

from typing import Any, Dict, Mapping, Tuple

import torch.nn as nn

from snn_interpreter.neurons.alpha import AlphaNeuron
from snn_interpreter.neurons.handler import NeuronHandler
from snn_interpreter.neurons.lapicque import LapicqueNeuron
from snn_interpreter.neurons.leaky import LeakyNeuron
from snn_interpreter.neurons.recurrent import RecurrentNeuron
from snn_interpreter.neurons.synaptic import SynapticNeuron

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
