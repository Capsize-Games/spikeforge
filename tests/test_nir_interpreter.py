"""Analytic and error-path tests for the independent NIR interpreter."""

import inspect
import math
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import torch

from spikeforge.nir_bridge import (
    api,
    interpreter,
    ops_linear,
    ops_neuron,
    ops_registry,
)
from spikeforge.nir_bridge.errors import UnsupportedNodeError
from spikeforge.nir_bridge.interpreter import NirInterpreter

pytest.importorskip("nir")

#: Hand-computed update parameters for the analytic LIF test.
TAU, R, V_THRESHOLD, V_RESET = 2.0, 3.0, 1.0, 0.0


class Mystery:
    """A node kind the interpreter deliberately does not implement."""


def _graph(
    nodes: Dict[str, Any], edges: List[Tuple[str, str]]
) -> Any:
    """Build an unchecked NIR graph from named nodes and edges."""
    return api.node_class("NIRGraph")(nodes, edges, type_check=False)


def _io_nodes() -> Tuple[Any, Any]:
    """Return fresh ``Input`` and ``Output`` nodes."""
    source = api.node_class("Input")({"input": None})
    sink = api.node_class("Output")({"output": None})
    return source, sink


def _analytic(
    state: float, current: float, tau: float, r: float, leak: float
) -> float:
    """Return one hand-computed zero-order-hold membrane update."""
    decay = math.exp(-ops_neuron.DT / tau)
    return leak + (state - leak) * decay + r * (1.0 - decay) * current


def _reference(
    spikes: torch.Tensor,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Return the hand-computed spike and membrane traces for ``spikes``."""
    states = [0.0, 0.0]
    spike_rows: List[List[float]] = []
    mem_rows: List[List[float]] = []
    for step in range(spikes.size(0)):
        row_spike: List[float] = []
        row_mem: List[float] = []
        for index in range(2):
            current = float(spikes[step, 0, index])
            value = _analytic(states[index], current, TAU, R, 0.0)
            fired = value > V_THRESHOLD
            states[index] = V_RESET if fired else value
            row_spike.append(1.0 if fired else 0.0)
            row_mem.append(value)
        spike_rows.append(row_spike)
        mem_rows.append(row_mem)
    return spike_rows, mem_rows


def _lif_graph() -> Any:
    """Return an ``Input -> LIF -> Output`` graph with known parameters."""
    lif = api.node_class("LIF")(
        np.array(TAU, np.float32),
        np.array(R, np.float32),
        np.array(0.0, np.float32),
        np.array(V_THRESHOLD, np.float32),
        np.array(V_RESET, np.float32),
    )
    source, sink = _io_nodes()
    return _graph(
        {"input": source, "lif": lif, "output": sink},
        [("input", "lif"), ("lif", "output")],
    )


def test_single_lif_matches_hand_computed_update() -> None:
    """A lone LIF node reproduces the analytic zero-order-hold recurrence."""
    spikes = torch.tensor(
        [[[0.4, 0.1]], [[0.2, 0.9]], [[0.7, 0.3]], [[0.0, 0.5]]]
    )
    result = NirInterpreter(_lif_graph()).run(spikes)
    spike_rows, mem_rows = _reference(spikes)
    expected_spikes = torch.tensor(spike_rows).unsqueeze(1)
    expected_mem = torch.tensor(mem_rows).unsqueeze(1)
    assert torch.allclose(result.spikes["lif"], expected_spikes)
    assert torch.allclose(result.membranes["lif"], expected_mem, atol=1e-6)


def test_interpreter_sources_never_mention_snntorch() -> None:
    """The independent execution path is free of any snnTorch reference."""
    modules = (interpreter, ops_neuron, ops_linear, ops_registry)
    for module in modules:
        assert "snntorch" not in inspect.getsource(module)


def test_unsupported_node_raises_typed_error() -> None:
    """An unknown node kind fails loudly, naming the node and its kind."""
    source, sink = _io_nodes()
    graph = _graph(
        {"input": source, "weird": Mystery(), "output": sink},
        [("input", "weird"), ("weird", "output")],
    )
    with pytest.raises(UnsupportedNodeError) as excinfo:
        NirInterpreter(graph)
    assert excinfo.value.kind == "Mystery"
    assert excinfo.value.name == "weird"


def _four_steps() -> torch.Tensor:
    """Return the analytic test's four-step, two-neuron input current."""
    return torch.tensor(
        [[[0.4, 0.1]], [[0.2, 0.9]], [[0.7, 0.3]], [[0.0, 0.5]]]
    )


def test_identity_post_node_is_byte_identical_and_sees_every_node() -> None:
    """An identity hook changes nothing and is called per computed node."""
    spikes = _four_steps()
    plain = NirInterpreter(_lif_graph()).run(spikes)
    seen: List[Tuple[str, str]] = []

    def identity(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Tuple[torch.Tensor, Any, Any]:
        seen.append((name, kind))
        return output, state, membrane

    hooked = NirInterpreter(_lif_graph(), post_node=identity).run(spikes)
    assert torch.equal(plain.readout, hooked.readout)
    assert torch.equal(plain.spikes["lif"], hooked.spikes["lif"])
    assert torch.equal(plain.membranes["lif"], hooked.membranes["lif"])
    assert seen == [("lif", "LIF"), ("output", "Output")] * spikes.size(0)


def test_post_node_state_is_what_the_next_step_carries() -> None:
    """Clearing the LIF state each step makes every update start at zero."""

    def clear(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Tuple[torch.Tensor, Any, Any]:
        if kind == "LIF":
            return output, torch.zeros_like(state), membrane
        return output, state, membrane

    spikes = _four_steps()
    hooked = NirInterpreter(_lif_graph(), post_node=clear).run(spikes)
    for step in range(spikes.size(0)):
        for index in range(2):
            current = float(spikes[step, 0, index])
            expected = _analytic(0.0, current, TAU, R, 0.0)
            got = float(hooked.membranes["lif"][step, 0, index])
            assert abs(got - expected) < 1e-6


def test_post_node_output_is_what_the_next_node_reads() -> None:
    """Silencing every LIF spike in the hook leaves the readout at zero."""

    def silence(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Tuple[torch.Tensor, Any, Any]:
        if kind == "LIF":
            return torch.zeros_like(output), state, membrane
        return output, state, membrane

    spikes = _four_steps()
    plain = NirInterpreter(_lif_graph()).run(spikes)
    hooked = NirInterpreter(_lif_graph(), post_node=silence).run(spikes)
    assert float(plain.readout.abs().sum()) > 0.0
    assert float(hooked.readout.abs().sum()) == 0.0
    assert float(hooked.spikes["lif"].sum()) == 0.0


def test_post_node_membrane_is_what_the_trace_records() -> None:
    """The recorded membrane is the hook's, not the value it was handed.

    The trace is written after the hook, so a hook that rewrites the
    membrane is visible in ``membranes``. Recording the pre-hook value
    instead would silently report an unquantized membrane beside a
    quantized one.
    """

    def blank(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Tuple[torch.Tensor, Any, Any]:
        if membrane is None:
            return output, state, membrane
        return output, state, torch.zeros_like(membrane)

    spikes = _four_steps()
    plain = NirInterpreter(_lif_graph()).run(spikes)
    hooked = NirInterpreter(_lif_graph(), post_node=blank).run(spikes)
    assert float(plain.membranes["lif"].abs().max()) > 0.0
    assert float(hooked.membranes["lif"].abs().max()) == 0.0
    assert torch.equal(plain.spikes["lif"], hooked.spikes["lif"])


def test_post_node_membrane_is_none_on_a_node_without_one() -> None:
    """A non-integrator node is handed ``None``, never a stand-in tensor."""
    seen: Dict[str, Any] = {}

    def record(
        name: str,
        kind: str,
        output: torch.Tensor,
        state: Any,
        membrane: Any,
    ) -> Tuple[torch.Tensor, Any, Any]:
        seen[kind] = membrane
        return output, state, membrane

    NirInterpreter(_lif_graph(), post_node=record).run(_four_steps())
    assert seen["Output"] is None
    assert torch.is_tensor(seen["LIF"])
