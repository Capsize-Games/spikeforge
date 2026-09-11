"""JSON-shaped request and response payloads for the headless service."""

from typing import Any, Dict, List, Mapping

import torch

from spikeforge.serving.prediction import Prediction
from spikeforge_serve.service import DEFAULT_SESSION


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


def frames_from(payload: Any) -> List[Any]:
    """Return the non-empty ``frames`` list carried by a request body."""
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("'frames' must be a non-empty list")
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
