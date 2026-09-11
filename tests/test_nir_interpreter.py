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
