"""Execute a target-ready graph on the Norse PyTorch simulator.

Norse supplies the neuron dynamics (``norse.LIF`` / ``norse.LI``) while the
stateless linear, convolution, pooling, threshold, and scale nodes reuse the
shared torch operations, exactly as Norse itself is pure PyTorch. The
compiler lowers a linear chain of layers; a graph that branches or merges is
refused with a named reason rather than executed in a guessed order.

The Norse leak ``p`` is derived as ``1 - exp(-1 / tau)`` so its discrete
recurrence matches the reference interpreter's zero-order-hold step, and the
node's input gain ``r * (1 - decay)`` is applied so any resistance is
honoured. An unsupported node on the chain makes ``run`` raise, which the
caller reports as a named error.
"""

from math import exp
from typing import Any, Dict, List, Tuple

import torch

from spikeforge.nir_bridge.ops_registry import OPS
from spikeforge_targets.backends import api, lowering
from spikeforge_targets.backends.errors import BackendUnavailableError
from spikeforge_targets.backends.result import (
    STATUS_OK,
    BackendResult,
)

#: Neuron kinds whose dynamics come from the Norse SDK.
NORSE_NEURONS = ("LIF", "LI")
#: Note attached to every completed Norse run.
NORSE_NOTE = "executed on the Norse PyTorch simulator"


def _decay(tau: float) -> float:
    """Return the discrete membrane decay for a time constant."""
    return exp(-1.0 / tau)


def _gain(node: Any, decay: float) -> float:
    """Return the input gain matching the reference zero-order-hold step."""
    return lowering.scalar(node.r) * (1.0 - decay)


def _reset(node: Any) -> Any:
    """Return the node's reset potential as a float, preserving ``None``."""
    return None if node.v_reset is None else lowering.scalar(node.v_reset)


def _lif_module(node: Any) -> Tuple[Any, float]:
    """Return the Norse LIF module and input gain for a NIR ``LIF`` node."""
    decay = _decay(lowering.scalar(node.tau))
    module = api.norse_lif(
        1.0 - decay,
        lowering.scalar(node.v_leak),
        lowering.scalar(node.v_threshold),
        _reset(node),
    )
    return module, _gain(node, decay)


def _li_module(node: Any) -> Tuple[Any, float]:
    """Return the Norse LI module and input gain for a NIR ``LI`` node."""
    decay = _decay(lowering.scalar(node.tau))
    module = api.norse_li(1.0 - decay, lowering.scalar(node.v_leak))
    return module, _gain(node, decay)


def _layer(name: str, node: Any) -> Tuple[Any, ...]:
    """Return the compiled layer tuple for one node."""
    kind = type(node).__name__
    if kind == "LIF":
        module, gain = _lif_module(node)
        return (name, kind, module, node, gain)
    if kind == "LI":
        module, gain = _li_module(node)
        return (name, kind, module, node, gain)
    return (name, kind, None, node, 1.0)


def _apply(
    layer: Tuple[Any, ...], value: Any, states: Dict[str, Any]
) -> Any:
    """Return one layer's output for ``value``."""
    name, kind, module, node, gain = layer
    if kind in NORSE_NEURONS:
        output, state = module(value * gain, states.get(name))
        states[name] = state
        return output
    return OPS[kind](node, value, None)[0]


def _record(
    name: str,
    kind: str,
    value: Any,
    spike_frames: Dict[str, List[Any]],
    mem_frames: Dict[str, List[Any]],
) -> None:
    """Append a neuron layer's output to its spike or membrane trace."""
    if kind == "LIF":
        spike_frames.setdefault(name, []).append(value)
    elif kind == "LI":
        mem_frames.setdefault(name, []).append(value)


def _step(
    layers: Tuple[Any, ...],
    states: Dict[str, Any],
    frame: Any,
    spike_frames: Dict[str, List[Any]],
    mem_frames: Dict[str, List[Any]],
) -> Any:
    """Thread one frame through the chain, returning the readout value."""
    value = frame
    readout = frame
    for layer in layers:
        kind = layer[1]
        if kind == "Input":
            continue
        if kind == "Output":
            readout = value
            continue
        value = _apply(layer, value, states)
        _record(layer[0], kind, value, spike_frames, mem_frames)
    return readout


def _stack(frames: Dict[str, List[Any]]) -> Dict[str, Any]:
    """Stack per-step frame lists into ``[T, ...]`` trace tensors."""
    return {name: torch.stack(items) for name, items in frames.items()}


def _execute(program: Any, spikes: Any) -> Tuple[Any, Any, Any, int]:
    """Run ``program`` over ``spikes`` and return traces, readout, steps."""
    layers = program["layers"]
    states: Dict[str, Any] = {}
    spike_frames: Dict[str, List[Any]] = {}
    mem_frames: Dict[str, List[Any]] = {}
    steps = int(spikes.size(0))
    total = None
    for index in range(steps):
        out = _step(layers, states, spikes[index], spike_frames, mem_frames)
        total = out if total is None else total + out
    readout = total / steps if total is not None else spikes.new_zeros(0)
    return _stack(spike_frames), _stack(mem_frames), readout, steps


class NorseBackend:
    """Compile and run a linear NIR chain on the Norse simulator."""

    name = "norse"

    def available(self) -> bool:
        """Return True when the ``norse`` package can be imported."""
        return api.module_available("norse")

    def compile(self, graph: Any, spec: Any) -> Any:
        """Lower ``graph`` to Norse layers, or raise a named reason."""
        if not self.available():
            raise BackendUnavailableError(self.name, "norse")
        chain = lowering.linear_chain(graph)
        return {"layers": tuple(_layer(name, node) for name, node in chain)}

    def run(self, compiled: Any, spikes: Any) -> BackendResult:
        """Execute ``compiled`` over ``spikes`` and return a result."""
        spike_map, mem_map, readout, steps = _execute(compiled, spikes)
        return BackendResult(
            target=self.name,
            status=STATUS_OK,
            steps=steps,
            readout=readout,
            spikes=spike_map,
            membranes=mem_map,
            notes=(NORSE_NOTE,),
        )
