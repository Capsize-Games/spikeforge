"""Outbound WebSocket message helpers."""

from server.session import Session


async def send_locked(ws, session: Session, message: dict):
    """Serialize sends so the stream task and loop never interleave."""
    async with session.lock:
        await ws.send_json(message)


async def send_initial(ws, session: Session, cfg):
    """Push the static payloads: ack, sample, recon, raster, status."""
    engine = session.engine
    await send_locked(ws, session, {
        "type": "config_ack", "payload": cfg.model_dump(),
    })
    await send_image(ws, session)
    await send_reconstruction(ws, session)
    await send_locked(ws, session, {
        "type": "raster", "payload": engine.raster(),
    })
    await send_status(ws, session, cfg)


async def send_image(ws, session):
    """Send the sample input image when the coding type has one."""
    image = session.engine.sample_image()
    if image is not None:
        await send_locked(ws, session, {"type": "image", "payload": image})


async def send_reconstruction(ws, session):
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


async def send_status(ws, session, cfg):
    """Send the run status summary to the client."""
    engine = session.engine
    await send_locked(ws, session, {
        "type": "status",
        "payload": {
            "coding": cfg.coding,
            "num_steps": engine.num_steps(),
            "target": engine.target_label(),
        },
    })


async def send_run_state(ws, session, running: bool, reason: str = ""):
    """Tell the client whether a stream is currently running."""
    await send_locked(ws, session, {
        "type": "run_state",
        "payload": {"running": running, "reason": reason},
    })


async def send_train_state(ws, session, running: bool, reason: str = ""):
    """Tell the client whether training is currently running."""
    await send_locked(ws, session, {
        "type": "train_state",
        "payload": {"running": running, "reason": reason},
    })


async def emit_frame(ws, session, engine, step):
    """Send one spike frame for the given step."""
    await send_locked(ws, session, {
        "type": "spike_frame",
        "payload": engine.spike_frame(step),
        "step": step,
    })
