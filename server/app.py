"""FastAPI application exposing the encoding engine over WebSocket."""

import asyncio
from typing import Dict

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.encoder import EncoderEngine
from server.schemas import ClientMessage
from server.session import Session
from server.web import mount_client

app = FastAPI(title="snn-interpreter server")

# Serve the built React app (no-op when client/dist is absent).
mount_client(app)

# One session per connected client.
_sessions: Dict[int, Session] = {}


async def _send_locked(ws: WebSocket, session: Session, message: dict):
    """Serialize sends so the stream task and loop never interleave."""
    async with session.lock:
        await ws.send_json(message)


async def _send_initial(ws: WebSocket, session: Session, cfg):
    """Push the static payloads: ack, sample, recon, raster, status."""
    engine = session.engine
    await _send_locked(ws, session, {
        "type": "config_ack", "payload": cfg.model_dump(),
    })
    await _send_image(ws, session)
    await _send_reconstruction(ws, session)
    await _send_locked(ws, session, {
        "type": "raster", "payload": engine.raster(),
    })
    await _send_status(ws, session, cfg)


async def _send_image(ws, session):
    """Send the sample input image when the coding type has one."""
    image = session.engine.sample_image()
    if image is not None:
        await _send_locked(ws, session, {"type": "image", "payload": image})


async def _send_reconstruction(ws, session):
    """Send gain=1 and low-gain averaged reconstructions for rate."""
    recon = session.engine.reconstruction()
    if recon is None:
        return
    await _send_locked(ws, session, {
        "type": "image", "payload": recon["gain1"], "kind": "recon_gain1",
    })
    await _send_locked(ws, session, {
        "type": "image", "payload": recon["low"], "kind": "recon_low",
    })


async def _send_status(ws, session, cfg):
    """Send the run status summary to the client."""
    engine = session.engine
    await _send_locked(ws, session, {
        "type": "status",
        "payload": {
            "coding": cfg.coding,
            "num_steps": engine.num_steps(),
            "target": engine.target_label(),
        },
    })


async def _send_run_state(ws, session, running: bool, reason: str = ""):
    """Tell the client whether a stream is currently running."""
    await _send_locked(ws, session, {
        "type": "run_state",
        "payload": {"running": running, "reason": reason},
    })


async def _stream_frames(ws: WebSocket, session: Session, cfg):
    """Push a spike_frame message for each time step, slowly."""
    engine = session.engine
    delay = max(0.02, cfg.interval_ms / 1000.0)
    if cfg.coding == "delta":
        await _emit_frame(ws, session, engine, 0)
    else:
        for step in range(engine.num_steps()):
            await _emit_frame(ws, session, engine, step)
            await asyncio.sleep(delay)
    await _send_run_state(ws, session, running=False, reason="finished")


async def _emit_frame(ws, session, engine, step):
    """Send one spike frame for the given step."""
    await _send_locked(ws, session, {
        "type": "spike_frame",
        "payload": engine.spike_frame(step),
        "step": step,
    })


async def _handle_run(ws, session: Session, cfg):
    """Start streaming frames as a cancellable background task."""
    if session.engine is None:
        session.set_engine(EncoderEngine(cfg))
    await _send_run_state(ws, session, running=True)
    await session.start(_stream_frames(ws, session, cfg))


async def _handle_stop(ws, session: Session):
    """Cancel the active stream immediately and notify the client."""
    await session.cancel()
    await _send_run_state(ws, session, running=False, reason="stopped")


async def _handle_configure(ws, session: Session, cfg):
    """Cancel any active stream and push fresh static payloads."""
    await session.cancel()
    session.set_engine(EncoderEngine(cfg))
    await _send_initial(ws, session, cfg)
    await _send_run_state(ws, session, running=False, reason="configured")


async def _dispatch(ws, session: Session, message: ClientMessage):
    """Route one client message to the matching handler."""
    if message.type == "configure":
        await _handle_configure(ws, session, message.config)
    elif message.type == "run":
        await _handle_run(ws, session, message.config)
    elif message.type == "stop":
        await _handle_stop(ws, session)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session_id = id(ws)
    session = Session()
    _sessions[session_id] = session
    try:
        while True:
            raw = await ws.receive_json()
            await _dispatch(ws, session, ClientMessage.model_validate(raw))
    except WebSocketDisconnect:
        await session.cancel()
        _sessions.pop(session_id, None)
    except Exception as exc:  # surface unexpected errors to the client
        await _send_locked(ws, session, {
            "type": "error", "payload": str(exc),
        })
        await _send_run_state(ws, session, running=False, reason="error")


@app.get("/health")
async def health():
    return {"status": "ok"}
