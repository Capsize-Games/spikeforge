"""Prove streaming/batch parity over one stateful inference session."""

from typing import Any, Dict, List

import torch

from spikeforge.serving.session import InferenceSession
from spikeforge.simulator.runner import run
from spikeforge.streaming.encoding import encode_windows

#: Default numerical tolerance for streaming-vs-batch parity.
PARITY_ATOL: float = 1e-5


def predict_batch(
    module: Any,
    windows: torch.Tensor,
    encode_spec: Any,
    *,
    delta_along_window: bool = False,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits for a whole batch (reference)."""
    module.eval()
    spikes = encode_windows(
        windows, encode_spec, delta_along_window=delta_along_window
    )
    with torch.no_grad():
        return run(module, spikes).logits.mean(dim=1)


def _stream_one_window(
    session: InferenceSession,
    window: torch.Tensor,
    encode_spec: Any,
    delta_along_window: bool,
) -> torch.Tensor:
    """Reset the session, then stream one window's spikes step by step."""
    spikes = encode_windows(
        window, encode_spec, delta_along_window=delta_along_window
    )
    session.reset()
    prediction = None
    for step in range(int(spikes.size(0))):
        prediction = session.step(spikes[step])
    if prediction is None:
        raise ValueError("an encoded window produced no timesteps")
    return prediction.mean_logits.mean(dim=1).reshape(-1)


def stream_logits(
    session: InferenceSession, windows: torch.Tensor, encode_spec: Any,
    *, delta_along_window: bool = False,
) -> torch.Tensor:
    """Return ``[B, C]`` mean logits streaming one window at a time.

    Reset per window through the stateful session -- exactly the serving
    path, so this loop and a whole-tensor ``run`` agree by construction.
    """
    rows = [
        _stream_one_window(
            session, windows[i:i + 1], encode_spec, delta_along_window
        )
        for i in range(int(windows.size(0)))
    ]
    if not rows:
        return torch.zeros(0, 0)
    return torch.stack(rows, dim=0)


def _argmax_labels(logits: torch.Tensor) -> List[int]:
    """Return the predicted class per row of ``logits``."""
    return [int(value) for value in logits.argmax(dim=-1).tolist()]


def _parity_fields(
    batch: torch.Tensor, stream: torch.Tensor, atol: float
) -> Dict[str, Any]:
    """Return the parity report fields for computed batch/stream logits."""
    difference = float((batch - stream).abs().max().item())
    return {
        "batch_logits": batch,
        "stream_logits": stream,
        "max_abs_diff": difference,
        "within_tolerance": difference <= float(atol),
        "batch_labels": _argmax_labels(batch),
        "stream_labels": _argmax_labels(stream),
    }


def parity_report(
    module: Any, session: InferenceSession, windows: torch.Tensor,
    encode_spec: Any, *, delta_along_window: bool = False,
    atol: float = PARITY_ATOL,
) -> Dict[str, Any]:
    """Return the streaming-vs-batch parity evidence for ``windows``."""
    batch = predict_batch(
        module, windows, encode_spec, delta_along_window=delta_along_window
    )
    stream = stream_logits(
        session, windows, encode_spec, delta_along_window=delta_along_window
    )
    return _parity_fields(batch, stream, atol)
