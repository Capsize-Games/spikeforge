"""Extract a NIR graph from an arbitrary third-party ``torch.nn.Module``.

The module is lifted into NIR through the isolated ``nirtorch`` wrapper in
:mod:`spikeforge.nir_bridge.api`, so the direct dependency stays in one
file. Modules ``nirtorch`` cannot map raise the typed
:class:`~spikeforge.nir_bridge.errors.UnsupportedNodeError` naming the
offending class; an unrecognised tracing operation raises the typed
:class:`~spikeforge.nir_bridge.errors.ExtractionError`. Nothing is ever
silently dropped. A successful graph is runnable by the independent
:class:`~spikeforge.nir_bridge.interpreter.NirInterpreter`, so a
third-party PyTorch model can be inspected and validated like any other.
"""

from typing import Any, Dict, Mapping, Optional

import torch

from spikeforge.nir_bridge import api, torch_map
from spikeforge.nir_bridge.errors import (
    ExtractionError,
    ExtractionExtraMissingError,
    UnsupportedNodeError,
)
from spikeforge.nir_bridge.exporter import graph_summary
from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge.nir_bridge.interpreter_result import InterpreterResult

#: Torch tracing failures that mean "this module cannot be represented".
_TRACE_FAILURES = (ValueError, RuntimeError, TypeError)


def _kind_name(error: KeyError) -> str:
    """Return the torch module class name recorded in a ``KeyError``."""
    item = error.args[0] if error.args else error
    return getattr(item, "__name__", str(item))


def _extract(module: Any, mapping: Mapping[Any, Any]) -> Any:
    """Call the isolated extractor, translating upstream failures."""
    try:
        return api.extract_graph(module, mapping)
    except KeyError as error:
        kind = _kind_name(error)
        raise UnsupportedNodeError(kind, kind) from None
    except _TRACE_FAILURES as error:
        raise ExtractionError(str(error)) from None


def extract(module: torch.nn.Module, module_map: Optional[Any] = None) -> Any:
    """Return the NIR graph extracted from ``module``.

    ``module_map`` overrides the default torch-module map. A missing
    ``nirtorch`` extra, an unmappable module, or an untraceable operation
    raises a typed error naming the reason instead of degrading.
    """
    if not api.nirtorch_available():
        raise ExtractionExtraMissingError("nir")
    graph = _extract(module, module_map or torch_map.NODE_MAP)
    if graph is None:
        raise ExtractionExtraMissingError("nir")
    return graph


def extract_summary(module: torch.nn.Module) -> Dict[str, Any]:
    """Return the JSON-able node/edge summary of ``module``'s NIR graph."""
    return graph_summary(extract(module))


def run_extracted(
    module: torch.nn.Module, spikes: torch.Tensor
) -> InterpreterResult:
    """Extract ``module`` and execute it on the independent interpreter."""
    return NirInterpreter(extract(module)).run(spikes)
