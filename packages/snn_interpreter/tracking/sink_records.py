"""Flatten a tracking record into the shapes external trackers accept."""

from typing import Any, Dict, Mapping


def _is_number(value: Any) -> bool:
    """Return True for a real number, excluding booleans."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def scalars(record: Mapping[str, Any]) -> Dict[str, float]:
    """Return numeric top-level fields plus the last history point."""
    data = {
        key: float(value)
        for key, value in record.items()
        if _is_number(value)
    }
    history = record.get("history") or []
    if history and isinstance(history[-1], Mapping):
        for key, value in history[-1].items():
            if _is_number(value):
                data[f"history.{key}"] = float(value)
    return data


def summary(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the JSON-able run summary a tracker stores as run config."""
    return {
        "config_hash": record.get("config_hash"),
        "seed": record.get("seed"),
        "tracking": dict(record.get("tracking") or {}),
        "determinism": dict(record.get("determinism") or {}),
    }
