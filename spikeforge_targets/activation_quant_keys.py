"""Which tensors of an evaluated node are quantizable, and under what key.

This is the one module that knows how a node's runtime values map onto report
keys, including the only node-kind-specific fact in the simulation: a
``CubaLIF`` carries its state as ``(synaptic current, membrane)``, so its two
halves are keyed apart while every other neuron's state is a lone membrane.
Keeping that here means the interpreter never learns a node's state layout
and the quantizer never re-derives one.

A node contributes at most three things: its output, which is an
``activation`` under the node's own name unless the node is one whose output
is exact or computed elsewhere; its carried state; and the membrane the
interpreter records. The carried state and the recorded trace are the same
membrane register read at two points in one step -- after the update, and
after the reset -- so both take the one ``<node>.membrane`` key. Keying them
apart would report one register as two, and calibrating on the reset value
alone would size the grid below the supra-threshold excursion the register
has to hold, clipping every spike-producing step.
"""

from typing import Any, List, Optional, Tuple

import torch

from spikeforge.nir_bridge.ops_registry import (
    DELAY,
    INPUT,
    OUTPUT,
    SPIKING_KINDS,
)
from spikeforge_targets.activation_quant import (
    TARGET_ACTIVATION,
    TARGET_MEMBRANE,
    is_float,
)

#: Node kinds whose output is never snapped: spikes are exact on any grid,
#: and the virtual nodes carry values that were computed elsewhere.
EXACT_KINDS = frozenset(SPIKING_KINDS) | {INPUT, OUTPUT, DELAY}
#: The node kind whose tuple state is ``(synaptic current, membrane)``.
CUBA = "CubaLIF"
#: Report and calibration key suffixes for a node's carried state.
MEMBRANE_SUFFIX = ".membrane"
CURRENT_SUFFIX = ".current"

#: ``(key, kind, tensor)`` triples naming one node's floating tensors.
Keyed = List[Tuple[str, str, torch.Tensor]]


def state_keys(name: str, kind: str, count: int) -> List[str]:
    """Return the report keys for a ``count``-tuple state of node ``name``."""
    if kind == CUBA and count == 2:
        return [name + CURRENT_SUFFIX, name + MEMBRANE_SUFFIX]
    return [f"{name}.state[{index}]" for index in range(count)]


def state_tensors(name: str, kind: str, state: Any) -> Keyed:
    """Return the floating tensors of ``state`` keyed as membranes."""
    if is_float(state):
        return [(name + MEMBRANE_SUFFIX, TARGET_MEMBRANE, state)]
    if isinstance(state, tuple):
        keys = state_keys(name, kind, len(state))
        return [
            (key, TARGET_MEMBRANE, item)
            for key, item in zip(keys, state)
            if is_float(item)
        ]
    return []


def node_tensors(
    name: str,
    kind: str,
    output: torch.Tensor,
    state: Any,
    membrane: Optional[torch.Tensor] = None,
) -> Keyed:
    """Return one evaluated node's floating tensors, keyed for a report."""
    items: Keyed = []
    if kind not in EXACT_KINDS and is_float(output):
        items.append((name, TARGET_ACTIVATION, output))
    items.extend(state_tensors(name, kind, state))
    if is_float(membrane):
        items.append((name + MEMBRANE_SUFFIX, TARGET_MEMBRANE, membrane))
    return items
