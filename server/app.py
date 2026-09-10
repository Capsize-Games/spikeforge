"""FastAPI application exposing the encoding engine over WebSocket."""

import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.downloads import manager
from server.handlers import dispatch
from server.messages import send_locked
from server.schemas import ClientMessage
from server.session import Session
from server.web import mount_client
from snn_interpreter.runtime import device as device_mod

app = FastAPI(title="snn-interpreter server")

# Warm the CUDA context once at startup so training/benchmarks don't stall.
device_mod.prime()

# Serve the built React app (no-op when client/dist is absent).
mount_client(app)

# One session per connected client.
_sessions: Dict[int, Session] = {}


async def _drain_training(ws: WebSocket, session: Session) -> None:
    """Forward worker-produced training messages to the socket."""
    while True:
        message = await session.training.queue.get()
        await send_locked(ws, session, message)


async def _dispatch_inbox(ws: WebSocket, session: Session) -> None:
    """Process queued client messages one at a time."""
    while True:
        message = await session.inbox.get()
        try:
            await dispatch(ws, session, message)
        except Exception as exc:  # keep the connection alive on bad input
            await send_locked(ws, session, {
                "type": "error", "payload": str(exc),
            })


async def _cleanup(
    session: Session,
    drain: asyncio.Task,
    worker: asyncio.Task,
    session_id: int,
) -> None:
    """Stop background work and forget the session."""
    session.training.stop()
    drain.cancel()
    worker.cancel()
    await session.cancel()
    _sessions.pop(session_id, None)


async def _serve(ws: WebSocket, session: Session) -> None:
    """Read client messages, giving downloads their own cancel path."""
    while True:
        raw = await ws.receive_json()
        try:
            message = ClientMessage.model_validate(raw)
        except Exception as exc:  # keep the connection alive on bad input
            await send_locked(ws, session, {
                "type": "error", "payload": str(exc),
            })
            continue
        # Cancel must bypass the queue so it lands while a download blocks the
        # dispatcher on its progress stream.
        if message.type == "cancel_download":
            manager.cancel()
            continue
        await session.inbox.put(message)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a client, run its session, and always clean up."""
    await ws.accept()
    session_id = id(ws)
    session = Session(asyncio.get_running_loop())
    _sessions[session_id] = session
    drain = asyncio.create_task(_drain_training(ws, session))
    worker = asyncio.create_task(_dispatch_inbox(ws, session))
    try:
        await _serve(ws, session)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # surface unexpected errors to the client
        await send_locked(ws, session, {
            "type": "error", "payload": str(exc),
        })
    finally:
        await _cleanup(session, drain, worker, session_id)


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe for container/orchestrator health checks."""
    return {"status": "ok"}
