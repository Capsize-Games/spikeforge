"""Typed decoding of the ``WS /v1/stream`` reply envelopes."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from spikeforge_clients.prediction import PredictionResult


@dataclass(frozen=True)
class StreamEvent:
    """One reply from the streaming endpoint (prediction, reset, session)."""

    type: str
    payload: Dict[str, Any]

    @classmethod
    def from_json(cls, message: Mapping[str, Any]) -> "StreamEvent":
        """Decode one ``{"type", "payload"}`` stream envelope."""
        payload = message.get("payload")
        return cls(
            type=str(message.get("type", "")),
            payload=dict(payload) if isinstance(payload, Mapping) else {},
        )

    @property
    def session_id(self) -> Optional[str]:
        """Return the reply's session id, when it carries one."""
        value = self.payload.get("session_id")
        return str(value) if value is not None else None

    @property
    def steps(self) -> Optional[int]:
        """Return the reply's step count, when it carries one."""
        value = self.payload.get("steps")
        return int(value) if value is not None else None

    @property
    def label(self) -> Optional[int]:
        """Return the predicted class index, for a prediction reply."""
        value = self.payload.get("label")
        return int(value) if value is not None else None

    @property
    def error(self) -> Optional[str]:
        """Return the error message, for an error reply."""
        if self.type != "error":
            return None
        value = self.payload.get("message") or self.payload.get("type")
        return str(value) if value is not None else "stream error"

    def prediction(self) -> Optional[PredictionResult]:
        """Decode a prediction reply, or return None for another type."""
        if self.type != "prediction":
            return None
        return PredictionResult.from_json(self.payload)

    def to_dict(self) -> Dict[str, Any]:
        """Return the event as its JSON-compatible mapping."""
        return {"type": self.type, "payload": self.payload}
