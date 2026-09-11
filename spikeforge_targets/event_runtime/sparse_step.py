"""A ``StageModule`` whose parameterised stages run as sparse operations.

The subclass only overrides the per-stage dispatch; the topology sweep,
input gathering, delayed edges, and neuron handlers are inherited unchanged,
so the sparse path cannot drift from the dense graph traversal. It adopts the
source module's submodules rather than copying them, so the sparse path runs
the very same weights the dense path does.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikeforge.neurons.contract import NeuronState
from spikeforge.topology.stage_module import StageModule
from spikeforge_targets.event_runtime import ops
from spikeforge_targets.event_runtime.counters import SynapticCounter
from spikeforge_targets.event_runtime.errors import UnsupportedKindError

#: Weight-free kinds the sparse path may run densely: no synaptic operations.
POINTWISE_KINDS: Tuple[str, ...] = (
    "flatten",
    "avgpool2d",
    "sumpool2d",
    "maxpool1d",
    "maxpool2d",
    "dropout",
    "add",
)


class SparseStageModule(StageModule):
    """Run a topology's parameterised stages over sparse spike events."""

    def __init__(
        self, source: StageModule, counter: Optional[SynapticCounter]
    ) -> None:
        """Adopt ``source``'s spec, dispatch tables, and submodules."""
        nn.Module.__init__(self)
        self._spec = source._spec
        self._order = source._order
        self._kinds = source._kinds
        self._inbound = source._inbound
        self._handlers = source._handlers
        self._counter = counter
        for name, child in source.named_children():
            self.add_module(name, child)

    def _run_stage(
        self,
        name: str,
        inputs: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, Optional[NeuronState]]:
        """Route one stage through its event-driven, neuron, or dense path."""
        kind = self._kinds[name]
        if name in self._handlers:
            return self._neuron(name, inputs, state)
        if kind in ops.SPARSE_KINDS:
            return self._sparse(name, kind, inputs)
        if kind in POINTWISE_KINDS:
            return self._pointwise(name, inputs)
        raise UnsupportedKindError(kind, name)

    def _neuron(
        self,
        name: str,
        inputs: torch.Tensor,
        state: Optional[NeuronState],
    ) -> Tuple[torch.Tensor, Optional[NeuronState]]:
        """Run a neuron stage and tally its elementwise accumulates."""
        output, new_state = super()._run_stage(name, inputs, state)
        self._count(name, 0, 0, output.numel())
        return output, new_state

    def _sparse(
        self, name: str, kind: str, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[NeuronState]]:
        """Run a linear/conv stage sparsely and tally SOP, MAC, and AC."""
        module = self.get_submodule(name)
        output, sop = ops.apply(kind, module, inputs)
        mac = output.numel() * ops.connected_inputs(kind, module)
        self._count(name, sop, mac, output.numel())
        return output, None

    def _pointwise(
        self, name: str, inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[NeuronState]]:
        """Run a weight-free stage densely and tally its accumulates."""
        output = self._apply_stage(name, inputs)
        self._count(name, 0, 0, output.numel())
        return output, None

    def _count(self, name: str, sop: int, mac: int, ac: int) -> None:
        """Record one stage's counts when a counter is attached."""
        if self._counter is not None:
            self._counter.record(name, sop, mac, ac)
