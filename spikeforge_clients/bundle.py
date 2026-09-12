"""A typed view of the ``GET /v1/bundle`` metadata payload."""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional


def _text(value: Any) -> Optional[str]:
    """Return ``value`` as text, or None when it is absent."""
    return None if value is None else str(value)


def _int(value: Any) -> Optional[int]:
    """Return ``value`` as an int, or None when it is absent."""
    return None if value is None else int(value)


@dataclass(frozen=True)
class BundleInfo:
    """The loaded bundle's self-describing metadata.

    The typed fields cover what a client needs to shape its input; ``raw``
    keeps the entire payload so an additive server key is never lost.
    """

    format: Optional[str]
    version: Optional[str]
    path: str
    topology: Optional[str]
    num_classes: Optional[int]
    input_size: Optional[int]
    encode_spec: Dict[str, Any]
    label_map: Dict[str, Any]
    expected_metrics: Dict[str, Any]
    raw: Dict[str, Any]

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "BundleInfo":
        """Decode a ``/v1/bundle`` response body."""
        return cls(
            format=_text(payload.get("format")),
            version=_text(payload.get("version")),
            path=str(payload.get("path", "")),
            topology=_text(payload.get("topology")),
            num_classes=_int(payload.get("num_classes")),
            input_size=_int(payload.get("input_size")),
            encode_spec=dict(payload.get("encode_spec") or {}),
            label_map=dict(payload.get("label_map") or {}),
            expected_metrics=dict(payload.get("expected_metrics") or {}),
            raw=dict(payload),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Return the full original payload (additive-safe)."""
        return dict(self.raw)
