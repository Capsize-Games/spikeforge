"""Handlers for the Phase 2 introspection WebSocket actions.

Mirrors :mod:`server.nir_handlers`: it lives outside :mod:`server.handlers`
so that module stays within the project's 250-line limit, and ``dispatch``
reaches :func:`dispatch_introspection` through the shared protocol router.
Every precondition failure emits the existing ``error`` message instead of
raising.
"""

from typing import Any, Optional, Tuple

from fastapi import WebSocket

from server.introspection_payloads import (
    benchmark_payload,
    metrics_payload,
    trajectory_payload,
)
from server.messages import send_introspection, send_locked
from server.schemas import ClientMessage, TrainConfig
from server.session import Session
from snn_interpreter.introspection.encoding import encoding_report
from snn_interpreter.introspection.surrogate import (
    list_surrogates,
    surrogate_curve,
)

#: Derivative-curve sampling defaults for the ``surrogate_curve`` action.
X_MIN = -2.0
X_MAX = 2.0
POINTS = 101
#: Execution mode required by the trajectory-capturing actions.
EDUCATIONAL_MODE = "educational"
#: Error returned when trajectory capture is requested in production mode.
EDUCATIONAL_ONLY = (
    "trajectory capture needs educational mode; "
    "switch the execution mode to educational"
)
#: Error returned when an image-only action is used on an event sample.
EVENT_UNSUPPORTED = (
    "encoding reconstruction decodes image codings; event datasets are "
    "already spike trains and carry no rate/latency signal to decode"
)


def _model_blocker(session: Session) -> Optional[str]:
    """Return why the active model cannot run, or None when it can."""
    if session.engine is None:
        return "no sample configured"
    if session.training.engine is None:
        return "no model loaded; train or load one"
    return None


def _active_mode(session: Session) -> str:
    """Return the active engine's execution mode.

    The default is ``educational`` so an engine that predates the mode field
    (or a test double) keeps the historical capture behaviour rather than
    silently failing; a real engine always declares its mode.
    """
    engine = session.training.engine
    return str(getattr(engine, "mode", EDUCATIONAL_MODE))


def _capture_blocker(session: Session) -> Optional[str]:
    """Return why trajectory capture cannot run, or None when it can."""
    blocker = _model_blocker(session)
    if blocker is not None:
        return blocker
    if _active_mode(session) != EDUCATIONAL_MODE:
        return EDUCATIONAL_ONLY
    return None


def _active(session: Session) -> Tuple[Any, Any, Any]:
    """Return the active ``(net, spec, spikes)`` triple."""
    engine = session.training.engine
    return engine.net, engine.spec, session.engine.spike_input()


async def _reject(ws: WebSocket, session: Session, reason: str) -> None:
    """Emit an error message carrying ``reason``."""
    await send_locked(ws, session, {"type": "error", "payload": reason})


async def handle_trajectory(ws: WebSocket, session: Session) -> None:
    """Emit bounded U[t]/I[t]/S[t] traces for the active model."""
    blocker = _capture_blocker(session)
    if blocker is not None:
        await _reject(ws, session, blocker)
        return
    await send_introspection(
        ws, session, "trajectory", trajectory_payload(*_active(session))
    )


async def handle_metrics(ws: WebSocket, session: Session) -> None:
    """Emit trajectory metrics for the active model."""
    blocker = _capture_blocker(session)
    if blocker is not None:
        await _reject(ws, session, blocker)
        return
    await send_introspection(
        ws, session, "metrics", metrics_payload(*_active(session))
    )


async def handle_encoding_report(ws: WebSocket, session: Session) -> None:
    """Emit the encoding report for the configured sample.

    Event modality has no image coding to decode, so it degrades to the
    same typed error channel instead of failing on a missing image.
    """
    if session.engine is None:
        await _reject(ws, session, "no sample configured")
        return
    engine = session.engine
    if getattr(engine, "modality", "image") == "event":
        await _reject(ws, session, EVENT_UNSUPPORTED)
        return
    report = encoding_report(engine.sample_tensor(), engine.encoder)
    await send_introspection(ws, session, "encoding_report", report)


async def handle_surrogates(ws: WebSocket, session: Session) -> None:
    """Emit the selectable surrogate-gradient names."""
    await send_introspection(ws, session, "surrogate_list", list_surrogates())


async def handle_surrogate_curve(
    ws: WebSocket, session: Session, name: Optional[str]
) -> None:
    """Emit the derivative curve for the named surrogate gradient."""
    if not name:
        await _reject(ws, session, "no surrogate selected")
        return
    try:
        curve = surrogate_curve(name, X_MIN, X_MAX, POINTS)
    except (KeyError, TypeError, ValueError) as exc:
        await _reject(ws, session, str(exc))
        return
    await send_introspection(ws, session, "surrogate_curve", curve)


async def handle_benchmark(
    ws: WebSocket, session: Session, cfg: TrainConfig
) -> None:
    """Emit the JSON-able report of a tiny benchmark run."""
    try:
        payload = benchmark_payload(cfg)
    except (KeyError, ValueError) as exc:
        await _reject(ws, session, str(exc))
        return
    await send_introspection(ws, session, "benchmark", payload)


async def dispatch_introspection(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route an introspection action to its handler."""
    if message.type == "trajectory":
        await handle_trajectory(ws, session)
    elif message.type == "metrics":
        await handle_metrics(ws, session)
    elif message.type == "encoding_report":
        await handle_encoding_report(ws, session)
    elif message.type == "surrogates":
        await handle_surrogates(ws, session)
    elif message.type == "surrogate_curve":
        await handle_surrogate_curve(ws, session, message.name)
    else:
        await handle_benchmark(ws, session, message.train)
