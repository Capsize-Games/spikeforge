"""Handler for the ``deploy_run`` backend-execution WebSocket action.

Mirrors :mod:`server.target_handlers`: it stays outside :mod:`server.handlers`
so that module keeps within the line limit. The capability view is unchanged
(it is still the ``deployment_report`` action); ``deploy_run`` adds the
*executed* view, compiling the target-ready graph and running it on the
backend. An unknown target is reported on the existing ``error`` channel and a
missing sample degrades to an error-shaped backend payload with a named
reason, never a raise.
"""

from typing import Any, Dict

from fastapi import WebSocket

from server.backend_payloads import backend_run_payload
from server.messages import send_backend_run, send_locked
from server.schemas import ClientMessage
from server.session import Session
from server.target_handlers import DEFAULT_TARGET, shaped_sample, target_pair
from snn_targets.registry import target_names


async def _resolve(
    ws: WebSocket, session: Session, message: ClientMessage
) -> Any:
    """Return the active ``(spec, module)`` pair or report an error."""
    try:
        return target_pair(session, message.train)
    except ValueError as exc:
        await send_locked(ws, session, {"type": "error", "payload": str(exc)})
        return None


async def handle_deploy_run(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Compile and run the active topology on the named backend."""
    target = message.name or DEFAULT_TARGET
    if target not in target_names():
        await send_locked(ws, session, {
            "type": "error", "payload": f"unknown target: {target!r}",
        })
        return
    pair = await _resolve(ws, session, message)
    if pair is None:
        return
    spec, module = pair
    payload: Dict[str, Any] = backend_run_payload(
        spec, module, target, shaped_sample(session, spec)
    )
    await send_backend_run(ws, session, payload)


async def dispatch_backend(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the backend-execution action."""
    await handle_deploy_run(ws, session, message)
