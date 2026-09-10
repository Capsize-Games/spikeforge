"""Resource-monitor statistics handler."""

from snn_interpreter.system_stats import snapshot

from server.messages import send_locked


async def handle_stats(ws, session):
    """Send a CPU/GPU memory snapshot for the resource monitor."""
    await send_locked(ws, session, {
        "type": "system_stats",
        "payload": snapshot(session.training.engine),
    })
