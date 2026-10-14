"""DoS bounds on ``WS /v1/stream`` single frames (issue #137).

The stream's single-frame path runs through the same size check as the
batch ``frames_from`` path, so an oversized stream frame is refused before
any tensor is allocated from it — ``predict`` never runs.
"""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from test_serve_app import _sample, _websocket, _write_bundle

from spikeforge_serve import create_app
from spikeforge_serve.payloads import frame_from
from spikeforge_serve.route_helpers import _stream_reply


class _NeverPredict:
    """Serving stub that fails the test if prediction is attempted."""

    def __init__(self) -> None:
        """Record that no prediction has been attempted yet."""
        self.calls: List[Any] = []

    def predict(self, frames: Any, **kwargs: Any) -> Any:
        """Record one call; a refused frame must never reach here."""
        self.calls.append((frames, kwargs))
        raise AssertionError("predict ran for a refused stream frame")


@pytest.fixture()
def bundle_path(tmp_path: Path) -> str:
    """Return the path to a freshly written test bundle."""
    path = tmp_path / "model.spkf"
    _write_bundle(path)
    return str(path)


def _refused(
    monkeypatch: pytest.MonkeyPatch, message: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the stream reply for ``message`` past a 16-element cap."""
    monkeypatch.setattr("spikeforge_serve.payloads.MAX_FRAME_ELEMENTS", 16)
    serving = _NeverPredict()
    _, reply = _stream_reply(serving, "session", message)
    assert serving.calls == []
    return reply


def test_oversized_stream_frame_refused_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A frame past the element cap errors without reaching predict."""
    reply = _refused(monkeypatch, {"frame": [[0.0] * 17]})
    assert reply["type"] == "error"
    assert "exceeds" in reply["payload"]["message"]


def test_deep_stream_frame_refused_before_predict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A nesting bomb errors without reaching predict."""
    frame: Any = [0.0]
    for _ in range(32):
        frame = [frame]
    reply = _refused(monkeypatch, {"frame": frame})
    assert reply["type"] == "error"
    assert "nesting" in reply["payload"]["message"]


def test_frame_from_rejects_missing_frame() -> None:
    """A stream message without ``frame`` is rejected, not misread."""
    with pytest.raises(ValueError, match="no 'frame'"):
        frame_from({})


def test_websocket_refuses_oversized_frame(bundle_path: str) -> None:
    """``WS /v1/stream`` answers an oversized frame with an error."""
    app = create_app(bundle_path)
    sent = asyncio.run(
        _websocket(
            app,
            "/v1/stream",
            [{"frame": _sample(7)}, {"frame": [[0.0] * 2_000_000]}],
        )
    )
    frames = [item for item in sent if item["type"] == "websocket.send"]
    replies = [json.loads(item["text"]) for item in frames]
    assert [reply["type"] for reply in replies] == [
        "prediction",
        "error",
    ]
    assert "exceeds" in replies[1]["payload"]["message"]
