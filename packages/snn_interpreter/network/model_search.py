"""Search the file-based checkpoint registry by its stored metadata.

Each result is a :func:`~snn_interpreter.network.model_store.list_models`
summary augmented with the ``accuracy`` read from the checkpoint history and
the stored ``manifest`` (or ``None`` for a legacy checkpoint). ``list_models``
itself is untouched, so existing callers keep their exact payload shape.
"""

from typing import Any, Dict, List, Mapping, Optional

from snn_interpreter.network import model_store


def _last_accuracy(history: List[Dict[str, Any]]) -> Optional[float]:
    """Return the newest non-null test accuracy, or None."""
    for point in reversed(history):
        value = point.get("test_accuracy")
        if value is not None:
            return float(value)
    return None


def _enrich(item: Dict[str, Any]) -> Dict[str, Any]:
    """Add accuracy and the raw manifest to one registry summary."""
    try:
        ckpt = model_store.load(item["name"])
    except Exception:
        ckpt = {}
    history = ckpt.get("history") or []
    return {
        **item,
        "accuracy": _last_accuracy(history),
        "manifest": ckpt.get("manifest"),
    }


def _selected(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the searchable field values for one registry record."""
    meta = record.get("meta") or {}
    return {
        "dataset": meta.get("dataset"),
        "topology": record.get("topology") or meta.get("topology"),
        "coding": record.get("coding") or meta.get("coding"),
        "input_mode": record.get("input_mode"),
        "device": meta.get("device"),
    }


def _equals(actual: Any, expected: Optional[str]) -> bool:
    """Return True when ``expected`` is unset or equals ``actual``."""
    return expected is None or actual == expected


def _name_matches(actual: str, needle: Optional[str]) -> bool:
    """Return True when ``needle`` is unset or a case-insensitive substring."""
    return needle is None or needle.lower() in actual.lower()


def _matches(
    record: Dict[str, Any],
    dataset: Optional[str],
    topology: Optional[str],
    coding: Optional[str],
    device: Optional[str],
    min_accuracy: Optional[float],
    name: Optional[str],
) -> bool:
    """Return True when ``record`` satisfies every supplied filter."""
    values = _selected(record)
    coding_ok = coding is None or coding in (
        values["coding"], values["input_mode"]
    )
    accuracy = record.get("accuracy")
    accuracy_ok = min_accuracy is None or (
        accuracy is not None and accuracy >= min_accuracy
    )
    return (
        _equals(values["dataset"], dataset)
        and _equals(values["topology"], topology)
        and _equals(values["device"], device)
        and coding_ok
        and accuracy_ok
        and _name_matches(record["name"], name)
    )


def search_models(
    dataset: Optional[str] = None,
    topology: Optional[str] = None,
    coding: Optional[str] = None,
    device: Optional[str] = None,
    min_accuracy: Optional[float] = None,
    name: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Return registry summaries matching every supplied filter.

    Unset filters are ignored. ``coding`` matches either the stored ``coding``
    or ``input_mode`` field, and ``name`` is a case-insensitive substring.
    """
    return [
        record
        for record in (_enrich(item) for item in model_store.list_models())
        if _matches(
            record, dataset, topology, coding, device, min_accuracy, name
        )
    ]
