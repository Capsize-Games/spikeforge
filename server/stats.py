"""Resource-monitor statistics handler."""

from fastapi import WebSocket

from server.messages import send_locked
from server.session import Session
from snn_interpreter.observability import metrics, persistence
from snn_interpreter.runtime.system_stats import snapshot


async def handle_stats(ws: WebSocket, session: Session) -> None:
    """Send a CPU/GPU memory snapshot plus in-process metrics.

    The ``metrics`` key is additive: existing consumers read ``cpu``/``gpu``/
    ``device`` unchanged and the dashboard may ignore the new key. The
    ``metrics_persisted`` flag and ``metrics_last_flush`` timestamp are also
    additive and report the opt-in persistence state without changing any
    existing key.
    """
    payload = snapshot(session.training.engine)
    payload["metrics"] = metrics.snapshot()
    status = persistence.status()
    payload["metrics_persisted"] = bool(status["enabled"])
    payload["metrics_last_flush"] = status["last_flush"]
    await send_locked(ws, session, {
        "type": "system_stats",
        "payload": payload,
    })
