"""Outbound WebSocket message helpers."""

from typing import Any, Dict

from fastapi import WebSocket

from server.protocol_version import PROTOCOL_VERSION
from server.schemas import EncodeConfig
from server.session import Session


async def send_locked(
    ws: WebSocket, session: Session, message: Dict[str, Any]
) -> None:
    """Serialize sends so the stream task and loop never interleave.

    Stamps the protocol version here, at the single funnel every outbound
    message passes through, so no helper can forget it.
    """
    message.setdefault("protocol_version", PROTOCOL_VERSION)
    async with session.lock:
        await ws.send_json(message)


async def send_initial(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Push the static payloads: ack, sample/event frame, raster, status."""
    engine = session.engine
    await send_locked(ws, session, {
        "type": "config_ack",
        "payload": {**cfg.model_dump(), "modality": engine.modality},
    })
    if engine.modality == "event":
        await send_event_frame(ws, session)
    else:
        await send_image(ws, session)
        await send_reconstruction(ws, session)
    await send_locked(ws, session, {
        "type": "raster", "payload": engine.raster(), "source": "input",
    })
    await send_status(ws, session, cfg)


async def send_image(ws: WebSocket, session: Session) -> None:
    """Send the sample input image when the coding type has one."""
    image = session.engine.sample_image()
    if image is not None:
        await send_locked(ws, session, {"type": "image", "payload": image})


async def send_event_frame(ws: WebSocket, session: Session) -> None:
    """Send the whole-sample ON/OFF frame as the sample image.

    The frame reuses the existing ``image`` message with an ``event_frame``
    kind instead of a dedicated type: it keeps one payload shape and one
    client field, while ``kind`` lets the viewer route it away from the
    static-sample slot.
    """
    await send_locked(ws, session, {
        "type": "image",
        "payload": session.engine.sample_frame(),
        "kind": "event_frame",
    })


async def send_reconstruction(ws: WebSocket, session: Session) -> None:
    """Send gain=1 and low-gain averaged reconstructions for rate."""
    recon = session.engine.reconstruction()
    if recon is None:
        return
    await send_locked(ws, session, {
        "type": "image", "payload": recon["gain1"], "kind": "recon_gain1",
    })
    await send_locked(ws, session, {
        "type": "image", "payload": recon["low"], "kind": "recon_low",
    })


async def send_status(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Send the run status summary to the client."""
    engine = session.engine
    await send_locked(ws, session, {
        "type": "status",
        "payload": {
            "coding": cfg.coding,
            "num_steps": engine.num_steps(),
            "target": engine.target_label(),
            "dataset": engine.dataset,
            "sample_index": engine.sample_index(),
            "true_label": engine.sample_label(),
            "modality": engine.modality,
        },
    })


async def send_run_state(
    ws: WebSocket, session: Session, running: bool, reason: str = ""
) -> None:
    """Tell the client whether a stream is currently running."""
    await send_locked(ws, session, {
        "type": "run_state",
        "payload": {"running": running, "reason": reason},
    })


async def send_train_state(
    ws: WebSocket, session: Session, running: bool, reason: str = "",
    mode: str = "",
) -> None:
    """Tell the client whether training is currently running."""
    await send_locked(ws, session, {
        "type": "train_state",
        "payload": {"running": running, "reason": reason,
                    "mode": mode or session.training.mode},
    })


async def send_download_state(
    ws: WebSocket, session: Session, state: Dict[str, Any]
) -> None:
    """Send a dataset-download progress snapshot to the client."""
    await send_locked(ws, session, {
        "type": "download_state", "payload": state,
    })


async def emit_frame(
    ws: WebSocket,
    session: Session,
    engine: Any,
    step: int,
    source: str = "input",
) -> None:
    """Send one spike frame for the given step and activity source."""
    await send_locked(ws, session, {
        "type": "spike_frame",
        "payload": engine.spike_frame(step),
        "step": step,
        "source": source,
    })


async def send_inference(
    ws: WebSocket, session: Session, result: Dict[str, Any]
) -> None:
    """Send the inference payload with private raster keys stripped."""
    payload = {k: v for k, v in result.items() if not k.startswith("_")}
    await send_locked(ws, session, {"type": "inference", "payload": payload})


async def send_activity(
    ws: WebSocket, session: Session, result: Dict[str, Any]
) -> None:
    """Send hidden/output rasters carried by an inference result."""
    rasters = result.get("_rasters", {})
    for source in ("hidden", "output"):
        raster = rasters.get(source)
        if raster is not None:
            await send_locked(ws, session, {
                "type": "raster", "payload": raster, "source": source,
            })


async def send_introspection(
    ws: WebSocket, session: Session, kind: str, payload: Any
) -> None:
    """Send an introspection payload under its message type ``kind``."""
    await send_locked(ws, session, {"type": kind, "payload": payload})


async def send_nir_graph(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send an NIR graph summary to the client."""
    await send_locked(ws, session, {"type": "nir_graph", "payload": payload})


async def send_nir_validation(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send a drift validation report to the client."""
    await send_locked(ws, session, {
        "type": "nir_validation", "payload": payload,
    })


async def send_target_list(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send the availability-annotated target registry to the client."""
    await send_locked(ws, session, {
        "type": "target_list", "payload": payload,
    })


async def send_model_search(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send filtered checkpoint registry records to the client."""
    await send_locked(ws, session, {
        "type": "model_search", "payload": payload,
    })


async def send_model_diff(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send a checkpoint metadata diff to the client."""
    await send_locked(ws, session, {
        "type": "model_diff", "payload": payload,
    })


async def send_deployment_report(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send a deployment report to the client."""
    await send_locked(ws, session, {
        "type": "deployment_report", "payload": payload,
    })


async def send_backend_run(
    ws: WebSocket, session: Session, payload: Dict[str, Any]
) -> None:
    """Send an executed backend run result to the client."""
    await send_locked(ws, session, {
        "type": "backend_run", "payload": payload,
    })
