"""An independent reference interpreter for exported NIR graphs.

This executes a ``nir.NIRGraph`` for ``T`` steps from its node parameters
alone using plain ``torch`` operations. It never imports or calls snnTorch,
the topology ``StageModule``, or the simulator runner, so comparing it with
the snnTorch forward pass is a genuine cross-check rather than a tautology.

Discretisation
--------------
NIR neurons are continuous-time nodes; this interpreter advances them with
the exact zero-order-hold form

    v[t] = v_leak + (v[t-1] - v_leak) * decay + R * (1 - decay) * I[t]

where ``decay = exp(-dt / tau)`` and one NIR time unit is one step
(``dt = 1``). ``Threshold`` and ``Scale`` are stateless pointwise nodes and
``Delay`` is a one-step stateful buffer, so a stage that feeds a reset back
into its own integrator still executes as a well-ordered graph: the delay
is evaluated at the start of each step from the previous step's value.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from snn_interpreter.nir_bridge.errors import UnsupportedNodeError
from snn_interpreter.nir_bridge.interpreter_result import InterpreterResult
from snn_interpreter.nir_bridge.ops_registry import (
    DELAY,
    INPUT,
    INTEGRATOR_KINDS,
    OPS,
    OUTPUT,
    SPATIAL_KINDS,
    SPIKING_KINDS,
    SUPPORTED_KINDS,
    children_map,
    drain,
    reduced_edges,
    stack,
    sum_inputs,
)

_Tensors = Dict[str, torch.Tensor]
_States = Dict[str, Any]
_Delays = Dict[str, Optional[torch.Tensor]]
_Frames = Dict[str, List[torch.Tensor]]


class NirInterpreter:
    """Execute an exported NIR graph step by step, without snnTorch."""

    def __init__(self, graph: Any) -> None:
        """Index ``graph`` and precompute ordering, delays and frame shape."""
        self._graph = graph
        self._nodes: Dict[str, Any] = dict(graph.nodes)
        self._kinds = {
            name: type(node).__name__ for name, node in self._nodes.items()
        }
        self._incoming = self._build_incoming()
        self._check_supported()
        self._order = self._execution_order()
        self._delays = self._build_delays()
        self._in_name = self._node_named(INPUT)
        self._out = self._node_named(OUTPUT)
        self._normalise = self._frame_normaliser()

    def _build_incoming(self) -> Dict[str, List[str]]:
        """Map every node to the sources of its incoming edges."""
        incoming: Dict[str, List[str]] = {name: [] for name in self._nodes}
        for source, target in self._graph.edges:
            incoming[target].append(source)
        return incoming

    def _check_supported(self) -> None:
        """Raise a typed error on any node kind the interpreter lacks."""
        for name, kind in self._kinds.items():
            if kind not in SUPPORTED_KINDS:
                raise UnsupportedNodeError(kind, name)

    def _is_delay(self, name: str) -> bool:
        """Return True when node ``name`` is a ``Delay`` node."""
        return self._kinds[name] == DELAY

    def _execution_order(self) -> List[str]:
        """Return a step order treating Delay nodes as step-start sources."""
        names = list(self._nodes)
        edges = reduced_edges(self._graph.edges, self._is_delay)
        children = children_map(names, edges)
        pending = dict.fromkeys(names, 0)
        for _, target in edges:
            pending[target] += 1
        ready = [name for name in names if pending[name] == 0]
        order = drain(ready, children, pending)
        if len(order) != len(names):
            raise ValueError("NIR graph cycle is not broken by a Delay node")
        return order

    def _first_consumer(self, name: str) -> Optional[str]:
        """Return the first node consuming ``name``'s output, if any."""
        for source, target in self._graph.edges:
            if source == name:
                return target
        return None

    def _frame_normaliser(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the per-step frame reshape matching the entry stage."""
        consumer = self._first_consumer(self._in_name)
        if consumer is not None and self._kinds[consumer] in SPATIAL_KINDS:
            return lambda frame: frame
        return lambda frame: frame.reshape(frame.size(0), -1)

    def _node_named(self, kind: str) -> str:
        """Return the name of the first node of ``kind``."""
        for name, node_kind in self._kinds.items():
            if node_kind == kind:
                return name
        raise ValueError(f"NIR graph has no {kind} node")

    def _build_delays(self) -> Dict[str, List[str]]:
        """Map every Delay node to the sources it buffers."""
        sources: Dict[str, List[str]] = {
            name: [] for name in self._nodes if self._is_delay(name)
        }
        for source, target in self._graph.edges:
            if target in sources:
                sources[target].append(source)
        return sources

    def run(self, spikes: torch.Tensor) -> InterpreterResult:
        """Execute the graph over a ``[T, ...]`` input spike train."""
        steps = int(spikes.size(0))
        state: _States = {}
        buffers: _Delays = dict.fromkeys(self._delays)
        spike_frames: _Frames = {}
        mem_frames: _Frames = {}
        total: Optional[torch.Tensor] = None
        for index in range(steps):
            frame = self._normalise(spikes[index])
            values, membranes = self._step(frame, state, buffers)
            self._record(values, membranes, spike_frames, mem_frames)
            total = self._seed(total, values[self._out])
        readout = total / steps if total is not None else spikes.new_zeros(0)
        return InterpreterResult(
            steps=steps,
            readout=readout,
            spikes=stack(spike_frames),
            membranes=stack(mem_frames),
        )

    @staticmethod
    def _seed(
        total: Optional[torch.Tensor], value: torch.Tensor
    ) -> torch.Tensor:
        """Return ``total`` plus ``value``, seeding on the first step."""
        return value if total is None else total + value

    def _step(
        self, frame: torch.Tensor, state: _States, buffers: _Delays
    ) -> Tuple[_Tensors, _Tensors]:
        """Evaluate every node once, then refresh the delay buffers."""
        values: _Tensors = {self._in_name: frame}
        membranes: _Tensors = {}
        for name in self._delays:
            values[name] = self._delay_value(name, buffers, frame)
        for name in self._order:
            kind = self._kinds[name]
            if kind not in (INPUT, DELAY):
                self._evaluate(name, kind, values, membranes, state)
        self._refresh(values, buffers)
        return values, membranes

    def _evaluate(
        self,
        name: str,
        kind: str,
        values: _Tensors,
        membranes: _Tensors,
        state: _States,
    ) -> None:
        """Evaluate node ``name``, recording a membrane when it has one."""
        output, new_state, membrane = self._apply(name, kind, values, state)
        values[name] = output
        if new_state is not None:
            state[name] = new_state
        if membrane is not None:
            membranes[name] = membrane

    def _refresh(self, values: _Tensors, buffers: _Delays) -> None:
        """Store each Delay node's current input for the next step."""
        for name, sources in self._delays.items():
            buffers[name] = sum_inputs([values[src] for src in sources])

    def _apply(
        self, name: str, kind: str, values: _Tensors, state: _States
    ) -> Any:
        """Apply node ``name``'s operation to its summed inputs."""
        sources = self._incoming[name]
        if not sources:
            raise ValueError(f"node {name!r} received no inputs")
        inputs = sum_inputs([values[source] for source in sources])
        return OPS[kind](self._nodes[name], inputs, state.get(name))

    def _delay_value(
        self, name: str, buffers: _Delays, frame: torch.Tensor
    ) -> torch.Tensor:
        """Return a delay's buffered output, zeros before its first step."""
        buffered = buffers.get(name)
        if buffered is None:
            return torch.zeros((), dtype=frame.dtype, device=frame.device)
        return buffered

    def _record(
        self,
        values: _Tensors,
        membranes: _Tensors,
        spike_frames: _Frames,
        mem_frames: _Frames,
    ) -> None:
        """Append this step's spike and membrane frames to the accumulators."""
        for name, kind in self._kinds.items():
            if kind in SPIKING_KINDS:
                spike_frames.setdefault(name, []).append(values[name])
            if kind in INTEGRATOR_KINDS:
                mem_frames.setdefault(name, []).append(membranes[name])
