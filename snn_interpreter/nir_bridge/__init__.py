"""Isolated bridge to the official NIR packages (Phase 1 scaffold).

Only :mod:`snn_interpreter.nir_bridge.api` imports ``nir``/``nirtorch``;
the rest of the project talks to those packages through this package. The
:func:`to_nir` exporter lifts a topology spec into a ``nir.NIRGraph``,
:func:`graph_summary` describes it as plain JSON data, and
:class:`~snn_interpreter.nir_bridge.interpreter.NirInterpreter` executes it
independently of snnTorch so :func:`validate` can report true numerical
drift.

Phase 5b adds an external import/export surface: :func:`save_graph` and
:func:`load_graph` persist a graph exactly, :func:`load_external` /
:func:`interpret_graph` / :func:`interpret_file` ingest and run a graph from
another framework, and :func:`roundtrip` proves the persisted artifact
reproduces the module's interpretation.
"""

from snn_interpreter.nir_bridge.api import (
    REPORT_KEYS,
    capability,
    node_class,
)
from snn_interpreter.nir_bridge.exporter import graph_summary, to_nir
from snn_interpreter.nir_bridge.ingest import (
    interpret_file,
    interpret_graph,
    load_external,
)
from snn_interpreter.nir_bridge.interpreter import NirInterpreter
from snn_interpreter.nir_bridge.interpreter_result import InterpreterResult
from snn_interpreter.nir_bridge.roundtrip import roundtrip
from snn_interpreter.nir_bridge.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    load_graph,
    save_graph,
)
from snn_interpreter.nir_bridge.validation_report import ValidationReport
from snn_interpreter.nir_bridge.validator import validate

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "REPORT_KEYS",
    "InterpreterResult",
    "NirInterpreter",
    "ValidationReport",
    "capability",
    "graph_summary",
    "interpret_file",
    "interpret_graph",
    "load_external",
    "load_graph",
    "node_class",
    "roundtrip",
    "save_graph",
    "to_nir",
    "validate",
]
