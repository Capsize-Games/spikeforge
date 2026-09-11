"""Route inbound client messages to stream/training/model actions."""

import asyncio
from typing import Optional

from fastapi import WebSocket

from server import animation
from server.engine_factory import build_encoder, ensure_dataset
from server.messages import (
    emit_frame,
    send_activity,
    send_inference,
    send_initial,
    send_locked,
    send_run_state,
    send_train_state,
)
from server.payloads import model_list_payload, model_loaded_payload
from server.protocol_handlers import PROTOCOL_ACTIONS, dispatch_protocol
from server.schemas import ClientMessage, EncodeConfig, TrainConfig
from server.session import Session
from server.stats import handle_stats
from snn_interpreter.network import model_store


async def stream_frames(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Push one spike_frame per step, looping until cancelled.

    With ``animate_hidden`` a hidden-layer frame is streamed per step too,
    sourced from the loaded model. When no model is present the reason is
    reported once and only the input frames stream, exactly as before.
    """
    engine = session.engine
    delay = max(0.02, cfg.interval_ms / 1000.0)
    steps = 1 if cfg.coding == "delta" else engine.num_steps()
    hidden = await animation.prepare(ws, session, cfg.animate_hidden)
    while True:
        for step in range(steps):
            await emit_frame(ws, session, engine, step)
            if hidden:
                await animation.emit_hidden_frame(ws, session, hidden, step)
            await asyncio.sleep(delay)


async def handle_run(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Start streaming frames as a cancellable background task."""
    if session.engine is None and not await build_encoder(ws, session, cfg):
        return
    await send_run_state(ws, session, running=True)
    await session.start(stream_frames(ws, session, cfg))


async def handle_stop(ws: WebSocket, session: Session) -> None:
    """Cancel the active stream immediately and notify the client."""
    await session.cancel()
    await send_run_state(ws, session, running=False, reason="stopped")


async def handle_configure(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Cancel any active stream and push fresh static payloads."""
    await session.cancel()
    if not await build_encoder(ws, session, cfg):
        return
    await send_initial(ws, session, cfg)
    await handle_run(ws, session, cfg)
    if session.training.engine is not None:
        await handle_infer(ws, session, cfg)


async def handle_select_sample(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Rebuild the engine for a new sample and replay the preview."""
    await session.cancel()
    if not await build_encoder(ws, session, cfg):
        return
    await send_initial(ws, session, cfg)
    await handle_run(ws, session, cfg)
    if session.training.engine is not None:
        await handle_infer(ws, session, cfg)


async def handle_infer(
    ws: WebSocket, session: Session, cfg: EncodeConfig
) -> None:
    """Infer on the displayed sample and emit inference + rasters."""
    blocker = _infer_blocker(session, cfg)
    if blocker is not None:
        await send_locked(ws, session, {"type": "error", "payload": blocker})
        return
    engine = session.engine
    spikes = engine.spike_input()
    result = session.training.infer(spikes, engine.sample_label())
    result["dataset_match"] = cfg.dataset == session.training.engine.dataset
    await send_inference(ws, session, result)
    await send_activity(ws, session, result)


def _infer_blocker(session: Session, cfg: EncodeConfig) -> Optional[str]:
    """Return why inference cannot run, or None when it can."""
    if session.engine is None:
        return "no sample configured"
    if session.training.engine is None:
        return "no model loaded; train or load one"
    if cfg.coding == "random":
        return "random coding carries no signal; pick another coding"
    return None


async def handle_train(
    ws: WebSocket,
    session: Session,
    cfg: TrainConfig,
    encode: Optional[EncodeConfig] = None,
) -> None:
    """Start training in a worker thread and stream its metrics."""
    if session.training.is_running:
        return
    if encode is None and "encode" in cfg.model_fields_set:
        encode = cfg.encode
    dataset = encode.dataset if encode is not None else cfg.dataset
    if not await ensure_dataset(ws, session, dataset):
        return
    session.training.start(cfg, encode)
    await send_train_state(ws, session, running=True, mode=cfg.mode)


async def handle_stop_train(ws: WebSocket, session: Session) -> None:
    """Signal the training worker to halt."""
    session.training.stop()
    await send_train_state(ws, session, running=False, reason="stopped")


async def handle_new_model(ws: WebSocket, session: Session) -> None:
    """Unload the current model so the next run starts from scratch."""
    session.training.clear()
    await send_locked(ws, session, {"type": "model_cleared", "payload": None})


async def handle_save_model(
    ws: WebSocket, session: Session, name: Optional[str]
) -> None:
    """Persist the current trained model to disk."""
    engine = session.training.engine
    if engine is None:
        await send_locked(ws, session, {
            "type": "error", "payload": "no trained model to save",
        })
        return
    path = session.training.save(name)
    await send_locked(ws, session, {
        "type": "model_saved", "payload": {"name": name, "path": path},
    })
    await handle_list_models(ws, session)


async def handle_list_models(ws: WebSocket, session: Session) -> None:
    """Send checkpoints, datasets, topologies, and neuron kinds."""
    await send_locked(ws, session, {
        "type": "model_list",
        "payload": model_list_payload(),
    })


async def handle_load_model(ws: WebSocket, session: Session,
                            name: Optional[str],
                            cfg: TrainConfig) -> None:
    """Load a checkpoint and make it the active training engine."""
    encode = (cfg.encode if "encode" in cfg.model_fields_set
              else session.encode_config)
    if encode is not None:
        session.set_config(encode)
    meta = model_store.load(name).get("meta", {}) if name else {}
    dataset = meta.get("dataset") or cfg.dataset
    if not await ensure_dataset(ws, session, dataset):
        return
    engine = session.training.adopt(name, cfg, encode)
    payload = model_loaded_payload(engine, name, encode, engine.evaluate())
    payload["history"] = model_store.load(name).get("history", [])
    await send_locked(ws, session, {
        "type": "model_loaded", "payload": payload,
    })


async def handle_delete_model(
    ws: WebSocket, session: Session, name: Optional[str]
) -> None:
    """Delete a saved checkpoint."""
    model_store.delete(name)
    await handle_list_models(ws, session)


async def _dispatch_model(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route model-management messages."""
    if message.type == "save_model":
        await handle_save_model(ws, session, message.name)
    elif message.type == "list_models":
        await handle_list_models(ws, session)
    elif message.type == "load_model":
        await handle_load_model(ws, session, message.name, message.train)
    elif message.type == "delete_model":
        await handle_delete_model(ws, session, message.name)
    elif message.type == "new_model":
        await handle_new_model(ws, session)
    elif message.type == "stats":
        await handle_stats(ws, session)


async def _dispatch_sample(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route sample-selection vs inference messages to their handler."""
    if message.type == "infer":
        await handle_infer(ws, session, message.config)
    else:
        await handle_select_sample(ws, session, message.config)


async def dispatch(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route one client message to the matching handler."""
    if message.type == "configure":
        await handle_configure(ws, session, message.config)
    elif message.type == "run":
        await handle_run(ws, session, message.config)
    elif message.type == "stop":
        await handle_stop(ws, session)
    elif message.type == "train":
        await handle_train(ws, session, message.train)
    elif message.type == "stop_train":
        await handle_stop_train(ws, session)
    elif message.type in PROTOCOL_ACTIONS:
        await dispatch_protocol(ws, session, message)
    elif message.type in ("select_sample", "infer"):
        await _dispatch_sample(ws, session, message)
    else:
        await _dispatch_model(ws, session, message)
