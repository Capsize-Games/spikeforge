"""FastAPI application exposing the encoding engine over WebSocket."""

import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from snn_interpreter import device as device_mod

from server.handlers import dispatch
from server.messages import send_locked
from server.schemas import ClientMessage
from server.session import Session
from server.web import mount_client

app = FastAPI(title="snn-interpreter server")

# Warm the CUDA context once at startup so training/benchmarks don't stall.
device_mod.prime()

# Serve the built React app (no-op when client/dist is absent).
mount_client(app)

# One session per connected client.
_sessions: Dict[int, Session] = {}


async def _drain_training(ws, session: Session):
    """Forward worker-produced training messages to the socket."""
    while True:
        message = await session.training.queue.get()
        await send_locked(ws, session, message)


async def _cleanup(session: Session, drain, session_id):
    """Stop training/streaming and forget the session."""
    session.training.stop()
    drain.cancel()
    await session.cancel()
    _sessions.pop(session_id, None)


async def _serve(ws: WebSocket, session: Session):
    """Receive-and-dispatch loop for one connection."""
    while True:
        raw = await ws.receive_json()
        try:
            await dispatch(ws, session, ClientMessage.model_validate(raw))
        except Exception as exc:  # keep the connection alive on bad input
            await send_locked(ws, session, {
                "type": "error", "payload": str(exc),
            })


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = id(ws)
    session = Session(asyncio.get_running_loop())
    _sessions[session_id] = session
    drain = asyncio.create_task(_drain_training(ws, session))
    try:
        await _serve(ws, session)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # surface unexpected errors to the client
        await send_locked(ws, session, {
            "type": "error", "payload": str(exc),
        })
    finally:
        await _cleanup(session, drain, session_id)


@app.get("/health")
async def health():
    return {"status": "ok"}
