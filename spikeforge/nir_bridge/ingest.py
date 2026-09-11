"""Import an external NIR graph and execute it independently.

Graphs may come from other frameworks, so the only requirement is the shared
NIR vocabulary: every node kind in ``targets.primitives.EMITTED_PRIMITIVES``
is understood, and anything else raises the typed
:class:`~spikeforge.nir_bridge.errors.UnsupportedNodeError` naming the
node instead of being skipped. Execution runs through the independent
:class:`~spikeforge.nir_bridge.interpreter.NirInterpreter`, so it never
touches snnTorch.
"""

from typing import Any

import torch

from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.interpreter_result import InterpreterResult
from spikeforge.nir_bridge.serialization import load_graph


def load_external(path: str) -> Any:
    """Return the external NIR graph stored at ``path``.

    This is the ingest entry point: it is exactly :func:`load_graph`, named to
    read as importing a graph produced elsewhere.
    """
    return load_graph(path)


def interpret_graph(graph: Any, spikes: torch.Tensor) -> InterpreterResult:
    """Execute an imported ``graph`` over a ``[T, ...]`` spike input."""
    return NirInterpreter(graph).run(spikes)


def interpret_file(path: str, spikes: torch.Tensor) -> InterpreterResult:
    """Load the external graph at ``path`` and execute it over ``spikes``."""
    return interpret_graph(load_external(path), spikes)
