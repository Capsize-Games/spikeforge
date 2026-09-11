"""Handler for the ``energy_report`` event-driven WebSocket action.

Mirrors :mod:`server.backend_handlers`: it stays outside :mod:`server.handlers`
so that module keeps within the line limit, and the protocol router reaches
:func:`dispatch_energy`. An unknown target is reported on the existing
``error`` channel and a missing sample degrades to a synthetic fixture inside
the payload builder, never a raise.
"""

from typing import Any, Optional

from fastapi import WebSocket

from server.energy_messages import send_energy_report
from server.energy_payloads import energy_payload
from server.messages import send_locked
from server.schemas import ClientMessage
from server.session import Session
from server.target_handlers import (
    DEFAULT_TARGET,
    shaped_sample,
    target_pair,
)
from snn_targets.registry import target_names


async def _resolve(
    ws: WebSocket, session: Session, message: ClientMessage
) -> Optional[Any]:
    """Return the active ``(spec, module)`` pair or report an error."""
    try:
        return target_pair(session, message.train)
    except ValueError as exc:
        await send_locked(ws, session, {"type": "error", "payload": str(exc)})
        return None


async def handle_energy_report(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit the energy/latency report for the active or configured topology."""
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
    payload = energy_payload(
        spec, module, target, shaped_sample(session, spec), message.sparse
    )
    await send_energy_report(ws, session, payload)


async def dispatch_energy(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the energy-report action."""
    await handle_energy_report(ws, session, message)
