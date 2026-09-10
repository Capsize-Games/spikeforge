"""Runnable per-step rendering of a :class:`TopologySpec`.

``StageModule`` is a plain ``nn.Module``: one submodule is registered per
parameterised or neuron stage **under the stage name**, so ``state_dict``
keys follow stage names (e.g. ``_fc1.weight``). The module owns no temporal
loop; the simulator drives :meth:`StageModule.step` once per time step.

State contract
--------------

``step(x, state)`` returns ``(outputs, state)`` where:

``outputs``  maps every stage name to its output tensor for this step,
             including the ``spec.input`` stage, whose output is ``x``.
``state``    maps each neuron stage name to its :data:`NeuronState` tuple
             (e.g. ``(mem,)`` for leaky, ``(syn, mem)`` for synaptic), plus
             the reserved ``PREV_KEY`` entry mapping stage name to the
             previous step's outputs and the reserved ``CURRENT_KEY`` entry
             mapping each neuron stage name to the merged inbound activation
             it received this step (its input current ``I[t]``). Delayed
             (feedback) edges read from the previous-step mapping, which is
             what threads recurrence without a time loop. Passing ``None``
             starts a fresh sequence. The reserved entries are additive: the
             ``(outputs, state)`` return shape and every caller are
             unchanged.
"""

from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch
import torch.nn as nn

from snn_interpreter.neurons.contract import NeuronState
from snn_interpreter.neurons.registry import NEURONS
from snn_interpreter.topology import stage_modules
from snn_interpreter.topology.kinds import PARAMETERLESS_KINDS
from snn_interpreter.topology.ordering import topological_order
from snn_interpreter.topology.spec import TopologySpec

#: Reserved state entry holding the previous step's stage outputs.
PREV_KEY = "__prev__"
#: Reserved state entry holding each neuron stage's input current this step.
CURRENT_KEY = "__current__"

StageOutputs = Dict[str, torch.Tensor]
StateMap = Dict[str, Any]


def _sum_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    total = tensors[0]
    for tensor in tensors[1:]:
        total = total + tensor
    return total


class StageModule(nn.Module):
    """Execute a :class:`TopologySpec` for a single time step."""

    def __init__(self, spec: TopologySpec) -> None:
        """Validate ``spec`` and register one submodule per stage."""
        super().__init__()
        spec.validate()
        self._spec = spec
        self._order = topological_order(spec.stages, spec.edges)
        self._kinds = {stage.name: stage.kind for stage in spec.stages}
        self._inbound = {name: spec.inbound(name) for name in self._kinds}
        self._handlers = {
            name: NEURONS[kind]
            for name, kind in self._kinds.items()
            if kind in NEURONS
        }
        for stage in spec.stages:
            module = stage_modules.module_for_stage(stage)
            if module is not None:
                self.add_module(stage.name, module)

    def step(
        self,
        x: torch.Tensor,
        state: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[StageOutputs, StateMap]:
        """Run one time step, returning ``(outputs, state)``."""
        current: StateMap = dict(state) if state else {}
        prev = current.get(PREV_KEY, {})
        outputs, updated, currents = self._sweep(x, prev, current)
        updated[PREV_KEY] = outputs
        updated[CURRENT_KEY] = currents
        return outputs, updated

    def _sweep(
        self,
        x: torch.Tensor,
        prev: Mapping[str, torch.Tensor],
        current: StateMap,
    ) -> Tuple[StageOutputs, StateMap, StageOutputs]:
        """Step every stage once, collecting outputs, states, and currents."""
        outputs: StageOutputs = {}
        updated: StateMap = {}
        currents: StageOutputs = {}
        for name in self._order:
            inputs = self._gather(name, x, outputs, prev)
            if name in self._handlers:
                currents[name] = inputs
            result = self._run_stage(name, inputs, current.get(name))
            output, new_state = result
            outputs[name] = output
            if new_state is not None:
                updated[name] = new_state
        return outputs, updated, currents

    def _run_stage(
        self,
        name: str,
        inputs: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, Optional[NeuronState]]:
        """Apply either a neuron handler or a plain module to ``inputs``."""
        handler = self._handlers.get(name)
        if handler is None:
            return self._apply_stage(name, inputs), None
        return handler.step(self.get_submodule(name), inputs, state)

    def _apply_stage(self, name: str, inputs: torch.Tensor) -> torch.Tensor:
        """Return the module output, or the summed input for merge stages.

        Named ``_apply_stage`` (not ``_apply``) so it does not shadow
        ``nn.Module._apply``, which device moves dispatch through.
        """
        if self._kinds[name] in PARAMETERLESS_KINDS:
            return inputs
        return self.get_submodule(name)(inputs)

    def _gather(
        self,
        name: str,
        x: torch.Tensor,
        outputs: StageOutputs,
        prev: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        """Sum the inputs of ``name``: forward outputs and delayed ones."""
        tensors: List[torch.Tensor] = []
        if name == self._spec.input:
            tensors.append(x)
        for edge in self._inbound[name]:
            if edge.delayed:
                if edge.source in prev:
                    tensors.append(prev[edge.source])
            else:
                tensors.append(outputs[edge.source])
        if not tensors:
            raise ValueError(f"stage {name!r} received no inputs")
        return _sum_tensors(tensors)
