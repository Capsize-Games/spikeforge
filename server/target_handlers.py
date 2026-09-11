"""Handlers for the deployment-target WebSocket actions.

Mirrors :mod:`server.nir_handlers`: it lives outside :mod:`server.handlers`
so that module stays within the project's 250-line limit, and ``dispatch``
reaches :func:`dispatch_targets` through the shared protocol router.

A report is always produced. An unknown target or topology is reported on the
existing ``error`` channel, and a validation failure degrades to a report
without the drift section (naming the reason in its notes) instead of raising,
because the capability classification is still useful on its own.
"""

from typing import Any, Dict, Optional, Tuple

from fastapi import WebSocket

from server.messages import (
    send_deployment_report,
    send_locked,
    send_target_list,
)
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from server.target_payloads import (
    deployment_report_payload,
    shaped_spikes,
    target_list_payload,
)
from snn_interpreter.targets.registry import target_names
from snn_interpreter.topology import registry

#: Target used when the client does not name one.
DEFAULT_TARGET = "reference"


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


def _shaped(session: Session, spec: Any) -> Optional[Any]:
    """Return the configured sample's shaped spikes, if a sample exists."""
    if session.engine is None:
        return None
    return shaped_spikes(session.engine.spike_input(), spec)


def target_pair(session: Session, cfg: TrainConfig) -> Tuple[Any, Any]:
    """Return the active engine's ``(spec, module)`` or a configured build."""
    return _target(session, cfg)


def shaped_sample(session: Session, spec: Any) -> Optional[Any]:
    """Return the configured sample's shaped spikes, or ``None``."""
    return _shaped(session, spec)


def _report(
    session: Session, spec: Any, module: Any, target: str
) -> Dict[str, Any]:
    """Return a report, dropping validation when it cannot be computed."""
    spikes = _shaped(session, spec)
    if spikes is None:
        return deployment_report_payload(spec, module, target)
    try:
        return deployment_report_payload(spec, module, target, spikes)
    except Exception as exc:  # the capability section is still truthful
        payload = deployment_report_payload(spec, module, target)
        payload["notes"].append(f"validation unavailable: {exc}")
        return payload


async def handle_targets(ws: WebSocket, session: Session) -> None:
    """Emit the availability-annotated target registry."""
    await send_target_list(ws, session, target_list_payload())


async def handle_deployment_report(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit a deployment report for the active or configured topology."""
    target = message.name or DEFAULT_TARGET
    if target not in target_names():
        await send_locked(ws, session, {
            "type": "error", "payload": f"unknown target: {target!r}",
        })
        return
    try:
        spec, module = _target(session, message.train)
    except ValueError as exc:
        await send_locked(ws, session, {"type": "error", "payload": str(exc)})
        return
    report = _report(session, spec, module, target)
    await send_deployment_report(ws, session, report)


async def dispatch_targets(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route the target list and deployment report actions."""
    if message.type == "targets":
        await handle_targets(ws, session)
    else:
        await handle_deployment_report(ws, session, message)
