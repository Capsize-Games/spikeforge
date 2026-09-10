"""Route inbound client messages to stream/training/model actions."""

import asyncio

from snn_interpreter import model_store
from snn_interpreter.datasets import catalog

from server.encoder import EncoderEngine
from server.messages import (
    emit_frame,
    send_activity,
    send_inference,
    send_initial,
    send_locked,
    send_run_state,
    send_train_state,
)
from server.payloads import model_loaded_payload
from server.schemas import ClientMessage
from server.session import Session
from server.stats import handle_stats


async def stream_frames(ws, session: Session, cfg):
    """Push a spike_frame message for each time step, slowly."""
    engine = session.engine
    delay = max(0.02, cfg.interval_ms / 1000.0)
    if cfg.coding == "delta":
        await emit_frame(ws, session, engine, 0)
    else:
        for step in range(engine.num_steps()):
            await emit_frame(ws, session, engine, step)
            await asyncio.sleep(delay)
    await send_run_state(ws, session, running=False, reason="finished")


async def handle_run(ws, session: Session, cfg):
    """Start streaming frames as a cancellable background task."""
    if session.engine is None:
        session.set_engine(EncoderEngine(cfg), cfg)
    await send_run_state(ws, session, running=True)
    await session.start(stream_frames(ws, session, cfg))


async def handle_stop(ws, session: Session):
    """Cancel the active stream immediately and notify the client."""
    await session.cancel()
    await send_run_state(ws, session, running=False, reason="stopped")


async def handle_configure(ws, session: Session, cfg):
    """Cancel any active stream and push fresh static payloads."""
    await session.cancel()
    session.set_engine(EncoderEngine(cfg), cfg)
    await send_initial(ws, session, cfg)
    await send_run_state(ws, session, running=False, reason="configured")
    if session.training.engine is not None:
        await handle_infer(ws, session, cfg)


async def handle_select_sample(ws, session: Session, cfg):
    """Rebuild the engine for a new sample and repaint the viewer."""
    await session.cancel()
    session.set_engine(EncoderEngine(cfg), cfg)
    await send_initial(ws, session, cfg)
    await send_run_state(ws, session, running=False, reason="sample")
    if session.training.engine is not None:
        await handle_infer(ws, session, cfg)


async def handle_infer(ws, session: Session, cfg):
    """Infer on the displayed sample and emit inference + rasters."""
    blocker = _infer_blocker(session, cfg)
    if blocker is not None:
        await send_locked(ws, session, {"type": "error", "payload": blocker})
        return
    engine = session.engine
    result = session.training.infer(engine.spike_input(), engine.sample_label())
    result["dataset_match"] = cfg.dataset == session.training.engine.dataset
    await send_inference(ws, session, result)
    await send_activity(ws, session, result)


def _infer_blocker(session: Session, cfg):
    """Return why inference cannot run, or None when it can."""
    if session.engine is None:
        return "no sample configured"
    if session.training.engine is None:
        return "no model loaded; train or load one"
    if cfg.coding == "random":
        return "random coding carries no signal; pick another coding"
    return None


async def handle_train(ws, session: Session, cfg, encode=None):
    """Start training in a worker thread and stream its metrics."""
    if session.training.is_running:
        return
    if encode is None and "encode" in cfg.model_fields_set:
        encode = cfg.encode
    session.training.start(cfg, encode)
    await send_train_state(ws, session, running=True)


async def handle_stop_train(ws, session: Session):
    """Signal the training worker to halt."""
    session.training.stop()
    await send_train_state(ws, session, running=False, reason="stopped")


async def handle_predict(ws, session: Session):
    """Predict sample digits with the current model, if trained."""
    engine = session.training.engine
    payload = (engine.predict_sample() if engine is not None
               else {"digits": [], "labels": []})
    await send_locked(ws, session, {"type": "prediction", "payload": payload})


async def handle_save_model(ws, session: Session, name):
    """Persist the current trained model to disk."""
    engine = session.training.engine
    if engine is None:
        await send_locked(ws, session, {
            "type": "error", "payload": "no trained model to save",
        })
        return
    path = engine.save(name)
    await send_locked(ws, session, {
        "type": "model_saved", "payload": {"name": name, "path": path},
    })
    await handle_list_models(ws, session)


async def handle_list_models(ws, session: Session):
    """Send the list of saved checkpoints plus dataset catalog."""
    await send_locked(ws, session, {
        "type": "model_list",
        "payload": {
            "models": model_store.list_models(),
            "datasets": catalog(),
        },
    })


async def handle_load_model(ws, session: Session, name, cfg):
    """Load a checkpoint and make it the active training engine."""
    encode = session.encode_config
    engine = session.training.adopt(name, cfg, encode)
    await send_locked(ws, session, {
        "type": "model_loaded",
        "payload": model_loaded_payload(
            engine, name, encode, engine.evaluate()),
    })


async def handle_delete_model(ws, session: Session, name):
    """Delete a saved checkpoint."""
    model_store.delete(name)
    await handle_list_models(ws, session)


async def _dispatch_model(ws, session: Session, message: ClientMessage):
    """Route model-management messages."""
    if message.type == "save_model":
        await handle_save_model(ws, session, message.name)
    elif message.type == "list_models":
        await handle_list_models(ws, session)
    elif message.type == "load_model":
        await handle_load_model(ws, session, message.name, message.train)
    elif message.type == "delete_model":
        await handle_delete_model(ws, session, message.name)
    elif message.type == "stats":
        await handle_stats(ws, session)


async def dispatch(ws, session: Session, message: ClientMessage):
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
    elif message.type == "predict":
        await handle_predict(ws, session)
    elif message.type in ("select_sample", "infer"):
        handler = handle_infer if message.type == "infer" else handle_select_sample
        await handler(ws, session, message.config)
    else:
        await _dispatch_model(ws, session, message)
