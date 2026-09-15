"""Isolated bridge to the official NIR packages (Phase 1 scaffold).

Only :mod:`spikeforge.nir_bridge.api` imports ``nir``/``nirtorch``;
the rest of the project talks to those packages through this package. The
:func:`to_nir` exporter lifts a topology spec into a ``nir.NIRGraph``,
:func:`graph_summary` describes it as plain JSON data, and
:class:`~spikeforge.nir_bridge.interpreter.NirInterpreter` executes it
independently of snnTorch so :func:`validate` can report true numerical
drift.

Phase 5b adds an external import/export surface: :func:`save_graph` and
:func:`load_graph` persist a graph exactly, :func:`load_external` /
:func:`interpret_graph` / :func:`interpret_file` ingest and run a graph from
another framework, and :func:`roundtrip` proves the persisted artifact
reproduces the module's interpretation.

Phase F3 adds :func:`extract`, :func:`extract_summary` and
:func:`run_extracted`, which lift an arbitrary third-party
``torch.nn.Module`` into NIR through ``nirtorch`` and run it on the
independent interpreter.
"""

from spikeforge.nir_bridge.api import (
    REPORT_KEYS,
    capability,
    node_class,
)
from spikeforge.nir_bridge.exporter import graph_summary, to_nir
from spikeforge.nir_bridge.extract import (
    extract,
    extract_summary,
    run_extracted,
)
from spikeforge.nir_bridge.ingest import (
    interpret_file,
    interpret_graph,
    load_external,
)
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.interpreter_result import InterpreterResult
from spikeforge.nir_bridge.post_node import PostNode
from spikeforge.nir_bridge.roundtrip import roundtrip
from spikeforge.nir_bridge.serialization import (
    FORMAT_NAME,
    FORMAT_VERSION,
    load_graph,
    save_graph,
)
from spikeforge.nir_bridge.validation_report import ValidationReport
from spikeforge.nir_bridge.validator import validate

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "REPORT_KEYS",
    "InterpreterResult",
    "NirInterpreter",
    "PostNode",
    "ValidationReport",
    "capability",
    "extract",
    "extract_summary",
    "graph_summary",
    "interpret_file",
    "interpret_graph",
    "load_external",
    "load_graph",
    "node_class",
    "roundtrip",
    "run_extracted",
    "save_graph",
    "to_nir",
    "validate",
]
