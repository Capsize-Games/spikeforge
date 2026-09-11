"""Handlers for the NIR export and validation WebSocket actions.

These live outside :mod:`server.handlers` so that module stays within the
project's 250-line limit; ``dispatch`` imports and routes to them.
"""

from typing import Any, Dict, Tuple

from fastapi import WebSocket

from server.messages import send_locked, send_nir_graph, send_nir_validation
from server.payloads import nir_graph_payload, nir_validation_payload
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from spikeforge.simulator import input_shape
from spikeforge.topology import registry


def _resolve(cfg: TrainConfig) -> Tuple[Any, Any]:
    """Build the configured topology as a fresh ``(spec, module)`` pair."""
    params: Dict[str, Any] = {
        "hidden": cfg.hidden,
        "beta": cfg.beta,
        **dict(cfg.topology_params),
    }
    return registry.build_topology(cfg.topology, params)


def _target(session: Session, cfg: TrainConfig) -> Tuple[Any, Any]:
    """Return the active engine's ``(spec, module)`` or a fresh build."""
    engine = session.training.engine
    if engine is not None:
        return engine.spec, engine.net
    return _resolve(cfg)


async def handle_nir_export(
    ws: WebSocket, session: Session, cfg: TrainConfig
) -> None:
    """Emit the NIR graph summary of the active or configured topology."""
    spec, module = _target(session, cfg)
    await send_nir_graph(ws, session, nir_graph_payload(spec, module))


async def handle_nir_validate(
    ws: WebSocket, session: Session, cfg: TrainConfig
) -> None:
    """Emit a drift report for the active or configured topology."""
    if session.engine is None:
        await send_locked(ws, session, {
            "type": "error", "payload": "no sample configured",
        })
        return
    spec, module = _target(session, cfg)
    spikes = input_shape.to_input_shape(session.engine.spike_input(), spec)
    report = nir_validation_payload(spec, module, spikes)
    await send_nir_validation(ws, session, report)


async def dispatch_nir(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the NIR export/validate actions to their handler."""
    if message.type == "nir_export":
        await handle_nir_export(ws, session, message.train)
    else:
        await handle_nir_validate(ws, session, message.train)
