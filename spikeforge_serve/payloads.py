"""JSON-shaped request and response payloads for the headless service."""

from typing import Any, Dict, Iterator, List, Mapping

import torch

from spikeforge.serving.prediction import Prediction
from spikeforge_serve.service import DEFAULT_SESSION

#: Largest number of frames one predict request may carry.
MAX_FRAMES = 512
#: Largest number of scalar elements one frame may carry.
MAX_FRAME_ELEMENTS = 1024 * 1024
#: Deepest nesting one frame may use (raw samples nest 3 deep).
MAX_FRAME_DEPTH = 16


def tensor_json(tensor: torch.Tensor) -> Dict[str, Any]:
    """Return a typed JSON view of ``tensor`` (dtype, shape, values)."""
    value = tensor.detach().cpu()
    dtype = str(value.dtype).replace("torch.", "", 1)
    return {
        "dtype": dtype,
        "shape": list(value.shape),
        "values": value.tolist(),
    }


def prediction_json(prediction: Prediction) -> Dict[str, Any]:
    """Return ``prediction`` as a JSON-ready typed payload."""
    return {
        "steps": int(prediction.steps),
        "label": int(prediction.label),
        "predicted": int(prediction.predicted),
        "logits": tensor_json(prediction.logits),
        "mean_logits": tensor_json(prediction.mean_logits),
        "class_totals": tensor_json(prediction.class_totals),
        "spikes": {
            name: tensor_json(value)
            for name, value in prediction.spikes.items()
        },
    }


def _item_size(value: Any, stack: List[Iterator[Any]]) -> int:
    """Count ``value``; queue a container for the iterative walk."""
    if isinstance(value, (str, bytes)):
        return len(value)
    if not isinstance(value, (list, tuple, Mapping)):
        return 1
    if len(stack) >= MAX_FRAME_DEPTH:
        raise ValueError("frame nesting exceeds the depth cap")
    if isinstance(value, Mapping):
        stack.append(iter(value.values()))
    else:
        stack.append(iter(value))
    return 0


def _frame_size(frame: Any) -> int:
    """Return ``frame``'s scalar count, rejecting an oversized frame."""
    total = 0
    stack: List[Iterator[Any]] = [iter([frame])]
    while stack:
        try:
            value = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        total += _item_size(value, stack)
        if total > MAX_FRAME_ELEMENTS:
            raise ValueError(
                "one frame exceeds the "
                f"{MAX_FRAME_ELEMENTS} element cap"
            )
    return total


def frames_from(payload: Any) -> List[Any]:
    """Return the bounded ``frames`` list carried by a request body.

    An over-count or oversized request is rejected here, before any
    tensor is allocated from its frames.
    """
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("'frames' must be a non-empty list")
    if len(frames) > MAX_FRAMES:
        raise ValueError(f"'frames' must hold at most {MAX_FRAMES}")
    for frame in frames:
        _frame_size(frame)
    return list(frames)


def encoded_flag(payload: Mapping[str, Any]) -> bool:
    """Return the request's ``encoded`` flag, defaulting to False."""
    return bool(payload.get("encoded", False))


def session_id_from(payload: Mapping[str, Any]) -> str:
    """Return the request's session id, defaulting to the shared one."""
    value = payload.get("session_id")
    if value is None:
        return DEFAULT_SESSION
    if not isinstance(value, str) or not value:
        raise ValueError("'session_id' must be a non-empty string")
    return value
