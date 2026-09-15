"""FastAPI application exposing the encoding engine over WebSocket."""

import asyncio
from typing import Any, Dict, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from server.auth import authorized
from server.bundle_routes import router as bundle_router
from server.downloads import manager
from server.handlers import dispatch
from server.hub_downloads import manager as hub_manager
from server.messages import send_locked
from server.protocol_version import PROTOCOL_MAJOR, PROTOCOL_VERSION
from server.schemas import ClientMessage
from server.session import Session
from server.web import mount_client
from spikeforge.runtime import device as device_mod

app = FastAPI(title="spikeforge server")

# Warm the CUDA context once at startup so training/benchmarks don't stall.
device_mod.prime()

app.include_router(bundle_router)

# Serve the built React app (no-op when client/dist is absent).
mount_client(app)

# One session per connected client.
_sessions: Dict[int, Session] = {}

#: Payload code returned when an inbound version is missing or mismatched.
VERSION_MISMATCH = "protocol_version_mismatch"


def _mismatch_payload(client: Optional[str], detail: str) -> Dict[str, Any]:
    """Return a version-mismatch payload naming ``client`` and ``detail``."""
    return {
        "code": VERSION_MISMATCH,
        "message": detail,
        "client": client,
        "server": PROTOCOL_VERSION,
    }


def _version_error(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return a mismatch payload, or None when the inbound version fits.

    Required from Phase 2 onward: a missing or MAJOR-mismatched version
    is rejected with ``payload.code = "protocol_version_mismatch"``.
    """
    incoming = raw.get("protocol_version")
    if incoming is None:
        return _mismatch_payload(
            None, "client message is missing protocol_version; server "
            f"protocol is {PROTOCOL_VERSION}",
        )
    major = str(incoming).split(".", 1)[0]
    if major == PROTOCOL_MAJOR:
        return None
    return _mismatch_payload(
        incoming, f"client protocol {incoming!r} is incompatible with "
        f"server protocol {PROTOCOL_VERSION}",
    )


async def _drain_training(ws: WebSocket, session: Session) -> None:
    """Forward worker-produced training messages to the socket."""
    while True:
        message = await session.training.queue.get()
        await send_locked(ws, session, message)


async def _drain_pipeline(ws: WebSocket, session: Session) -> None:
    """Forward worker-produced pipeline messages to the socket."""
    while True:
        message = await session.pipeline.queue.get()
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
    pipeline_drain: asyncio.Task,
    worker: asyncio.Task,
    session_id: int,
) -> None:
    """Stop background work and forget the session."""
    session.training.stop()
    session.pipeline.stop()
    drain.cancel()
    pipeline_drain.cancel()
    worker.cancel()
    await session.cancel()
    _sessions.pop(session_id, None)


async def _check_version(
    ws: WebSocket, session: Session, raw: Dict[str, Any]
) -> bool:
    """Report a version mismatch and return whether one occurred."""
    version_error = _version_error(raw)
    if version_error is None:
        return False
    await send_locked(ws, session, {
        "type": "error", "payload": version_error,
    })
    return True


async def _parse_or_reject(
    ws: WebSocket, session: Session, raw: Dict[str, Any]
) -> Optional[Any]:
    """Return the parsed message, or None after reporting a parse error."""
    try:
        return ClientMessage.model_validate(raw)
    except Exception as exc:  # keep the connection alive on bad input
        await send_locked(ws, session, {
            "type": "error", "payload": str(exc),
        })
        return None


async def _handle_hub_cancel(ws: WebSocket, session: Session) -> None:
    """Cancel the in-flight hub download and report its new state."""
    hub_manager.cancel()
    await send_locked(ws, session, {
        "type": "hub_download_state",
        "payload": hub_manager.snapshot(),
    })


async def _serve(ws: WebSocket, session: Session) -> None:
    """Read client messages, giving downloads their own cancel path."""
    while True:
        raw = await ws.receive_json()
        if await _check_version(ws, session, raw):
            continue
        message = await _parse_or_reject(ws, session, raw)
        if message is None:
            continue
        # Cancels must bypass the queue so they land while a download blocks
        # the dispatcher on its progress stream.
        if message.type == "cancel_download":
            manager.cancel()
            continue
        if message.type == "hub_cancel":
            await _handle_hub_cancel(ws, session)
            continue
        await session.inbox.put(message)


async def _spawn_workers(
    ws: WebSocket, session: Session
) -> Any:
    """Start the training/pipeline drains and the inbox dispatcher."""
    return (
        asyncio.create_task(_drain_training(ws, session)),
        asyncio.create_task(_drain_pipeline(ws, session)),
        asyncio.create_task(_dispatch_inbox(ws, session)),
    )


async def _run_session(
    ws: WebSocket, session: Session, drain: asyncio.Task,
    pipeline_drain: asyncio.Task, worker: asyncio.Task, session_id: int,
) -> None:
    """Serve the session, then always clean up its background work."""
    try:
        await _serve(ws, session)
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # surface unexpected errors to the client
        await send_locked(ws, session, {
            "type": "error", "payload": str(exc),
        })
    finally:
        await _cleanup(session, drain, pipeline_drain, worker, session_id)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    """Accept a client, run its session, and always clean up.

    A token-gated connection is rejected before ``accept()`` so the
    handshake fails outright rather than opening then closing.
    """
    if not authorized(
        ws.query_params.get("token"), ws.headers.get("authorization")
    ):
        await ws.close(code=1008)
        return
    await ws.accept()
    session_id = id(ws)
    session = Session(asyncio.get_running_loop())
    _sessions[session_id] = session
    drain, pipeline_drain, worker = await _spawn_workers(ws, session)
    await _run_session(
        ws, session, drain, pipeline_drain, worker, session_id
    )


@app.get("/health")
async def health() -> Dict[str, str]:
    """Liveness probe for container/orchestrator health checks."""
    return {"status": "ok"}
