"""Cross-library ingest of externally built NIR graphs."""

import math
from typing import Any, List, Tuple

import numpy as np
import pytest
import torch

from snn_interpreter.nir_bridge import api
from snn_interpreter.nir_bridge.errors import UnsupportedNodeError
from snn_interpreter.nir_bridge.ingest import (
    interpret_file,
    interpret_graph,
    load_external,
)
from snn_interpreter.nir_bridge.serialization import save_graph

pytest.importorskip("nir")

#: Hand-computed LIF parameters for the external graph.
TAU = 2.0
R = 1.0
THRESHOLD = 0.5

#: A fixed spike train shaped ``[T, batch, features]``.
_SPIKES = torch.tensor(
    [
        [[0.9, 0.1], [0.2, 0.8]],
        [[0.0, 0.3], [0.7, 0.4]],
        [[1.0, 1.0], [0.0, 0.0]],
    ]
)


def _external_graph() -> Any:
    """Return an ``Input -> Affine -> LIF -> Output`` graph built with nir."""
    nodes = {
        "input": api.node_class("Input")({"input": np.array([2])}),
        "affine": api.node_class("Affine")(
            np.eye(2, dtype=np.float32), np.zeros(2, "f4")
        ),
        "lif": api.node_class("LIF")(
            np.array(TAU, np.float32),
            np.array(R, np.float32),
            np.array(0.0, np.float32),
            np.array(THRESHOLD, np.float32),
            np.array(0.0, np.float32),
        ),
        "output": api.node_class("Output")({"output": np.array([2])}),
    }
    edges = [("input", "affine"), ("affine", "lif"), ("lif", "output")]
    graph_cls = api.node_class("NIRGraph")
    return graph_cls(nodes, edges, type_check=False)


def _analytic_lif(spikes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the hand-computed leaky update for an identity affine input."""
    decay = math.exp(-1.0 / TAU)
    state = torch.zeros(spikes.shape[1], spikes.shape[2])
    spike_rows: List[torch.Tensor] = []
    mem_rows: List[torch.Tensor] = []
    for step in range(spikes.shape[0]):
        membrane = state * decay + (1.0 - decay) * spikes[step]
        spike = (membrane > THRESHOLD).to(spikes.dtype)
        state = torch.where(spike > 0, torch.zeros_like(membrane), membrane)
        spike_rows.append(spike)
        mem_rows.append(membrane)
    return torch.stack(spike_rows), torch.stack(mem_rows)


def test_external_graph_runs_through_interpreter(tmp_path: Any) -> None:
    """A directly-built graph loads and reproduces its analytic run."""
    path = str(tmp_path / "external.nir.json")
    save_graph(_external_graph(), path)
    result = interpret_file(path, _SPIKES)
    expected_spikes, expected_mem = _analytic_lif(_SPIKES)
    assert torch.equal(result.spikes["lif"], expected_spikes)
    assert torch.allclose(result.membranes["lif"], expected_mem, atol=1e-6)
    assert torch.allclose(
        result.readout, expected_spikes.mean(dim=0), atol=1e-6
    )


def test_in_memory_and_loaded_graphs_agree(tmp_path: Any) -> None:
    """The in-memory ingest path matches the reloaded file path exactly."""
    path = str(tmp_path / "external.nir.json")
    graph = _external_graph()
    save_graph(graph, path)
    in_memory = interpret_graph(graph, _SPIKES)
    loaded = interpret_graph(load_external(path), _SPIKES)
    assert torch.equal(in_memory.readout, loaded.readout)
    assert torch.equal(in_memory.spikes["lif"], loaded.spikes["lif"])
    assert set(load_external(path).nodes) == set(graph.nodes)


def _unsupported_graph() -> Any:
    """Return a graph whose integrator the interpreter does not support."""
    nodes = {
        "input": api.node_class("Input")({"input": np.array([2])}),
        "integrator": api.node_class("I")(np.array([1.0], np.float32)),
        "output": api.node_class("Output")({"output": np.array([2])}),
    }
    edges = [("input", "integrator"), ("integrator", "output")]
    graph_cls = api.node_class("NIRGraph")
    return graph_cls(nodes, edges, type_check=False)


def test_imported_unsupported_node_raises_named_error(tmp_path: Any) -> None:
    """An unsupported imported node fails loudly, naming node and kind."""
    path = str(tmp_path / "unsupported.nir.json")
    save_graph(_unsupported_graph(), path)
    with pytest.raises(UnsupportedNodeError) as excinfo:
        interpret_graph(load_external(path), torch.rand(3, 1, 2))
    assert excinfo.value.kind == "I"
    assert excinfo.value.name == "integrator"
