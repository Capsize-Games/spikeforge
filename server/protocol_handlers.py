"""Route the NIR and introspection WebSocket actions to their dispatchers.

Keeping this thin router separate lets :mod:`server.handlers` stay within
the 20-line function limit: one branch delegates every NIR and introspection
action here, and this module fans them out to :mod:`server.nir_handlers` and
:mod:`server.introspection_handlers`.
"""

from fastapi import WebSocket

from server.backend_handlers import dispatch_backend
from server.energy_handlers import dispatch_energy
from server.hub_handlers import dispatch_hub
from server.introspection_handlers import dispatch_introspection
from server.model_handlers import dispatch_registry
from server.nir_handlers import dispatch_nir
from server.pipeline_handlers import PIPELINE_ACTIONS, dispatch_pipeline
from server.schemas import ClientMessage
from server.session import Session
from server.target_handlers import dispatch_targets

#: Actions handled by the NIR, introspection, target, registry, hub, and
#: pipeline modules.
PROTOCOL_ACTIONS = frozenset({
    "nir_export", "nir_validate",
    "trajectory", "metrics", "encoding_report",
    "surrogates", "surrogate_curve", "benchmark",
    "targets", "deployment_report", "deploy_run", "energy_report",
    "model_search", "model_diff",
    "hub_list", "hub_search", "hub_download", "hub_cancel",
    "hub_inspect", "hub_import",
}) | PIPELINE_ACTIONS


async def dispatch_protocol(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route a NIR, target, registry, hub, or pipeline action."""
    if message.type.startswith("hub_"):
        await dispatch_hub(ws, session, message)
    elif message.type in ("nir_export", "nir_validate"):
        await dispatch_nir(ws, session, message)
    elif message.type in ("targets", "deployment_report"):
        await dispatch_targets(ws, session, message)
    elif message.type == "deploy_run":
        await dispatch_backend(ws, session, message)
    elif message.type == "energy_report":
        await dispatch_energy(ws, session, message)
    elif message.type in ("model_search", "model_diff"):
        await dispatch_registry(ws, session, message)
    elif message.type in PIPELINE_ACTIONS:
        await dispatch_pipeline(ws, session, message)
    else:
        await dispatch_introspection(ws, session, message)
