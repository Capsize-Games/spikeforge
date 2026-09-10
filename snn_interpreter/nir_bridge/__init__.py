"""Isolated bridge to the official NIR packages (Phase 1 scaffold).

Only :mod:`snn_interpreter.nir_bridge.api` imports ``nir``/``nirtorch``;
the rest of the project talks to those packages through this package. The
:func:`to_nir` exporter lifts a topology spec into a ``nir.NIRGraph``,
:func:`graph_summary` describes it as plain JSON data, and
:class:`~snn_interpreter.nir_bridge.interpreter.NirInterpreter` executes it
independently of snnTorch so :func:`validate` can report true numerical
drift.
"""

from snn_interpreter.nir_bridge.api import (
    REPORT_KEYS,
    capability,
    node_class,
)
from snn_interpreter.nir_bridge.exporter import graph_summary, to_nir
from snn_interpreter.nir_bridge.interpreter import NirInterpreter
from snn_interpreter.nir_bridge.interpreter_result import InterpreterResult
from snn_interpreter.nir_bridge.validation_report import ValidationReport
from snn_interpreter.nir_bridge.validator import validate

__all__ = [
    "REPORT_KEYS",
    "InterpreterResult",
    "NirInterpreter",
    "ValidationReport",
    "capability",
    "graph_summary",
    "node_class",
    "to_nir",
    "validate",
]
