"""Resolve, describe, and emit to the configured external tracking sink.

The local manifest is always written first by the caller, so a tracker outage
never loses a run: this module only decides whether a requested sink is usable
and, if so, forwards the record. An absent backend resolves to a named no-op
whose ``reason`` becomes the manifest's honest explanation.
"""

from typing import Any, Dict, List, Mapping, Optional

from snn_interpreter.tracking.sink import Sink
from snn_interpreter.tracking.tensorboard_sink import TensorBoardSink
from snn_interpreter.tracking.wandb_sink import WandBSink

_FACTORIES = {
    "tensorboard": TensorBoardSink,
    "wandb": WandBSink,
}


class NoopSink:
    """A named sink that never forwards, used when a backend is absent."""

    def __init__(self, name: str, reason: str) -> None:
        """Record the requested name and why nothing will be forwarded."""
        self._name = name
        self._reason = reason

    @property
    def name(self) -> str:
        """Return the requested sink name."""
        return self._name

    @property
    def reason(self) -> str:
        """Return why nothing will be forwarded."""
        return self._reason

    def available(self) -> bool:
        """Report that a no-op sink is never available."""
        return False

    def log(self, record: Mapping[str, Any]) -> bool:
        """Do nothing and report that nothing was forwarded."""
        return False


def sink_names() -> List[str]:
    """Return the names of the known external sinks."""
    return sorted(_FACTORIES)


def resolve(requested: Optional[str]) -> Sink:
    """Return the requested sink, or a named no-op when it is unavailable."""
    if not requested:
        return NoopSink("none", "no sink requested")
    factory = _FACTORIES.get(requested)
    if factory is None:
        return NoopSink(requested, f"unknown sink '{requested}'")
    sink = factory()
    if not sink.available():
        return NoopSink(
            requested, f"sink '{requested}' backend is not installed"
        )
    return sink


def describe(requested: Optional[str]) -> Dict[str, Any]:
    """Return the manifest ``tracking`` block for a requested sink."""
    sink = resolve(requested)
    return {
        "requested": requested or None,
        "active": sink.available(),
        "reason": sink.reason,
    }


def emit(record: Mapping[str, Any]) -> bool:
    """Forward ``record`` to the manifest's requested sink; never raises."""
    block = record.get("tracking") or {}
    sink = resolve(block.get("requested"))
    if not sink.available():
        return False
    try:
        return bool(sink.log(record))
    except Exception:
        return False
