"""JSON-able payloads for the model-registry WebSocket actions.

Mirrors :mod:`server.target_payloads`: the search reuses
:func:`snn_interpreter.network.model_search.search_models` so the socket
surface matches the ``records`` CLI exactly, and the diff reuses
:func:`snn_interpreter.network.model_diff.checkpoint_diff`.
"""

from typing import Any, Dict

from server.schemas import ModelQuery
from snn_interpreter.network import model_search


def search_payload(query: ModelQuery) -> Dict[str, Any]:
    """Return the registry records matching ``query``'s filters."""
    return {
        "models": model_search.search_models(
            dataset=query.dataset,
            topology=query.topology,
            coding=query.coding,
            device=query.device,
            min_accuracy=query.min_accuracy,
            name=query.name,
        )
    }
