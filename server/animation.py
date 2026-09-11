"""Per-step hidden-layer animation frames and their availability message.

The frames come from the loaded model's hidden stage. Without a model there
is nothing honest to animate, so availability and its named reason are
reported to the client rather than a fabricated frame; with no model the
input-frame stream stays exactly as it was.
"""

from typing import Any, List, Optional, Tuple

import torch
from fastapi import WebSocket

from server.messages import send_locked
from server.session import Session
from spikeforge.network.hidden_frames import hidden_frame_series

#: Named reasons the hidden animation cannot run.
NO_MODEL = "no model loaded; train or load one"
NO_SAMPLE = "no sample configured"


def _on_device(net: Any, spikes: torch.Tensor) -> torch.Tensor:
    """Move ``spikes`` onto the network's parameter device."""
    param = next(net.parameters(), None)
    return spikes if param is None else spikes.to(param.device)


def hidden_series(session: Session) -> Tuple[Optional[List[Any]], str]:
    """Return the per-step hidden frames and a reason when unavailable."""
    engine = session.training.engine
    if engine is None:
        return None, NO_MODEL
    sample = session.engine
    if sample is None:
        return None, NO_SAMPLE
    try:
        spikes = _on_device(engine.net, sample.spike_input())
        return hidden_frame_series(engine.net, spikes), ""
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


async def send_animation_state(
    ws: WebSocket, session: Session, available: bool, reason: str
) -> None:
    """Report whether the hidden animation is available, and why not."""
    await send_locked(ws, session, {
        "type": "animation_state",
        "payload": {"available": available, "reason": reason,
                    "source": "hidden"},
    })


async def emit_hidden_frame(
    ws: WebSocket, session: Session, frames: List[Any], step: int
) -> None:
    """Send one step's hidden-layer frame as a ``spike_frame``."""
    index = min(max(int(step), 0), len(frames) - 1)
    await send_locked(ws, session, {
        "type": "spike_frame",
        "payload": frames[index],
        "step": index,
        "source": "hidden",
    })


async def prepare(
    ws: WebSocket, session: Session, animate: bool
) -> List[Any]:
    """Return the hidden frame series, reporting availability once."""
    if not animate:
        return []
    frames, reason = hidden_series(session)
    await send_animation_state(ws, session, frames is not None, reason)
    return frames or []
