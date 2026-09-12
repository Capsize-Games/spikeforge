"""Typed decoding of the service's JSON prediction payloads.

Every ``/v1/predict`` response and every stream ``prediction`` envelope is
decoded into the dataclasses here, so a caller reads ``.label`` or
``.logits.values`` instead of walking dictionaries. The shapes mirror
``spikeforge_serve.payloads.prediction_json`` exactly.
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


@dataclass(frozen=True)
class TensorValue:
    """A typed tensor: its dtype, shape, and nested numeric values."""

    dtype: str
    shape: Tuple[int, ...]
    values: Any

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "TensorValue":
        """Decode a ``tensor_json`` payload."""
        return cls(
            dtype=str(payload.get("dtype", "")),
            shape=tuple(int(dim) for dim in payload.get("shape", [])),
            values=payload.get("values"),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the tensor as its JSON-compatible mapping."""
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "values": self.values,
        }


@dataclass(frozen=True)
class PredictionResult:
    """One frame's readout plus the session's cumulative evidence."""

    steps: int
    label: int
    predicted: int
    logits: TensorValue
    mean_logits: TensorValue
    class_totals: TensorValue
    spikes: Dict[str, TensorValue]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PredictionResult":
        """Decode a ``prediction_json`` payload."""
        spikes = payload.get("spikes") or {}
        return cls(
            steps=int(payload["steps"]),
            label=int(payload["label"]),
            predicted=int(payload["predicted"]),
            logits=TensorValue.from_json(payload["logits"]),
            mean_logits=TensorValue.from_json(payload["mean_logits"]),
            class_totals=TensorValue.from_json(payload["class_totals"]),
            spikes={
                str(name): TensorValue.from_json(value)
                for name, value in spikes.items()
            },
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the prediction as its JSON-compatible mapping."""
        return {
            "steps": self.steps,
            "label": self.label,
            "predicted": self.predicted,
            "logits": self.logits.to_dict(),
            "mean_logits": self.mean_logits.to_dict(),
            "class_totals": self.class_totals.to_dict(),
            "spikes": {
                name: value.to_dict()
                for name, value in self.spikes.items()
            },
        }


@dataclass(frozen=True)
class PredictResponse:
    """A ``/v1/predict`` response: its session, step count, and predictions."""

    session_id: str
    steps: int
    predictions: Tuple[PredictionResult, ...]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PredictResponse":
        """Decode a ``/v1/predict`` response body."""
        return cls(
            session_id=str(payload["session_id"]),
            steps=int(payload["steps"]),
            predictions=tuple(
                PredictionResult.from_json(item)
                for item in payload.get("predictions", [])
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the response as its JSON-compatible mapping."""
        return {
            "session_id": self.session_id,
            "steps": self.steps,
            "predictions": [
                item.to_dict() for item in self.predictions
            ],
        }


@dataclass(frozen=True)
class ResetResult:
    """A ``/v1/reset`` response: the session id and its zeroed step count."""

    session_id: str
    steps: int

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "ResetResult":
        """Decode a ``/v1/reset`` response body."""
        return cls(
            session_id=str(payload["session_id"]),
            steps=int(payload["steps"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the result as its JSON-compatible mapping."""
        return {"session_id": self.session_id, "steps": self.steps}
