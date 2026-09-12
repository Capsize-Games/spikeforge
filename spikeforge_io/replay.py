"""Replay a recorded stream into a sink or into ``spikeforge-serve``'s stream.

Two entry points:

* :func:`replay` feeds each adapted frame to any callable sink, so a recorded
  file can drive the ``encode`` function, an
  :class:`~spikeforge.serving.session.InferenceSession`, or a test double.
* :func:`stream_messages` / :func:`replay_to_stream` build and send the exact
  message shape the ``WS /v1/stream`` endpoint accepts (``{"frame": ...}`` with
  an optional leading ``{"reset": true}``), so an adapter replays a recorded
  stream into the live service without a bespoke transport.

Neither function imports the serve or client distributions: the protocol is a
plain dict, and the caller supplies the ``send`` callable, so
``spikeforge-io`` stays dependency-light.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import torch

from spikeforge_io.adapters import StreamAdapter

#: A callable that receives one frame tensor (or message dict).
Sink = Callable[[Any], Any]


@dataclass(frozen=True)
class ReplayReport:
    """What one replay sent: the source, frame count, and any replies."""

    source: str
    frames: int
    windows: bool = False
    replies: Sequence[Any] = field(default_factory=tuple)

    def to_dict(self) -> Dict[str, Any]:
        """Return the report as a JSON-able mapping."""
        return {
            "source": self.source,
            "frames": int(self.frames),
            "windows": bool(self.windows),
            "replies": len(self.replies),
        }


def _frames(adapter: StreamAdapter, window_spec: Any) -> List[torch.Tensor]:
    """Return the adapter's replay frames as a list of tensors.

    Without a ``window_spec`` each frame is one raw ``[D]`` sample; with one,
    each frame is a normalized ``[L, D]`` window.
    """
    if window_spec is None:
        return list(adapter.stream())
    return list(adapter.windows(window_spec))


def replay(
    adapter: StreamAdapter,
    sink: Sink,
    *,
    window_spec: Any = None,
) -> ReplayReport:
    """Feed every frame of ``adapter`` to ``sink`` and return the report.

    ``sink`` receives one frame at a time (a ``[D]`` sample, or an ``[L, D]``
    window when ``window_spec`` is given); its return value is ignored. This is
    the general hook an :class:`~spikeforge.serving.session.InferenceSession`,
    the ``encode`` function, or a test double binds to.
    """
    frames = _frames(adapter, window_spec)
    for frame in frames:
        sink(frame)
    return ReplayReport(
        source=adapter.source,
        frames=len(frames),
        windows=window_spec is not None,
    )


def stream_messages(
    adapter: StreamAdapter,
    *,
    window_spec: Any = None,
    encoded: bool = False,
    session_id: Optional[str] = None,
    reset: bool = False,
) -> Iterator[Dict[str, Any]]:
    """Yield the ``/v1/stream`` messages that replay ``adapter``.

    The first message is a ``{"reset": true}`` envelope when ``reset`` is set,
    then one ``{"frame": ...}`` message per sample (or per normalized window
    when ``window_spec`` is given). The shape matches
    :func:`spikeforge_serve.app._stream_reply` exactly.
    """
    if reset:
        yield _message({"reset": True}, session_id)
    for frame in _frames(adapter, window_spec):
        yield _message(
            {"frame": frame.tolist(), "encoded": bool(encoded)}, session_id
        )


def _message(
    body: Dict[str, Any], session_id: Optional[str]
) -> Dict[str, Any]:
    """Return ``body`` with an optional session id folded in."""
    if session_id:
        body["session_id"] = session_id
    return body


def replay_to_stream(
    adapter: StreamAdapter,
    send: Sink,
    *,
    window_spec: Any = None,
    encoded: bool = False,
    session_id: Optional[str] = None,
    reset: bool = False,
) -> ReplayReport:
    """Replay ``adapter`` into ``/v1/stream`` through ``send``.

    ``send`` takes one protocol message and returns the service reply (for a
    WebSocket, ``send`` would serialize and await the response). The replies
    are collected into the returned :class:`ReplayReport`.
    """
    frames = 0
    replies: List[Any] = []
    for message in stream_messages(
        adapter,
        window_spec=window_spec,
        encoded=encoded,
        session_id=session_id,
        reset=reset,
    ):
        if "frame" in message:
            frames += 1
        replies.append(send(message))
    return ReplayReport(
        source=adapter.source,
        frames=frames,
        windows=window_spec is not None,
        replies=tuple(replies),
    )
