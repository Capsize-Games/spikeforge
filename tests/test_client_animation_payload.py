"""Server-side hidden-frame animation: payload shape and honest gating."""

import asyncio
import contextlib
from typing import Any, Dict, Iterator, List

import torch

from server import animation
from server.handlers import stream_frames
from server.schemas import EncodeConfig
from server.session import Session
from spikeforge.network import hidden_frames
from spikeforge.topology import presets
from spikeforge.topology.builder import build_module

_SPEC = presets.fc_legacy(
    hidden=4, beta=0.5, num_classes=3, input_size=784
)
_STEPS = 3
_INTERVAL_MS = 20


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


class _Net:
    """A minimal training-engine stand-in exposing its ``net``."""

    def __init__(self, module: Any) -> None:
        self.net = module

    def parameters(self) -> Iterator[torch.Tensor]:
        return iter(self.net.parameters())


class _Sample:
    """A minimal encoder-engine stand-in serving a spike input."""

    def __init__(self, spikes: torch.Tensor) -> None:
        self._spikes = spikes

    def spike_input(self) -> torch.Tensor:
        return self._spikes

    def num_steps(self) -> int:
        return _STEPS

    def spike_frame(self, step: int) -> Any:
        return [[float(step)]]


def _spikes() -> torch.Tensor:
    return torch.rand(_STEPS, 1, 784)


def _session(module: Any, spikes: torch.Tensor, animate: bool) -> Session:
    cfg = EncodeConfig(animate_hidden=animate)
    session = Session(asyncio.get_running_loop())
    session.set_engine(_Sample(spikes), cfg)
    session.training._engine = _Net(module)
    return session


def test_hidden_frame_series_is_one_row_per_step() -> None:
    """The series has one single-row frame per step over the hidden width."""
    module = build_module(_SPEC)
    frames = hidden_frames.hidden_frame_series(module, _spikes())
    assert len(frames) == _STEPS
    assert len(frames[0]) == 1
    assert len(frames[0][0]) == 4


def test_hidden_series_names_a_missing_model() -> None:
    """Without a loaded model the series is unavailable, by name."""

    async def _run() -> Any:
        session = Session(asyncio.get_running_loop())
        session.set_engine(_Sample(_spikes()), EncodeConfig())
        return animation.hidden_series(session)

    frames, reason = asyncio.run(_run())
    assert frames is None
    assert reason == animation.NO_MODEL


def test_prepare_without_animation_sends_nothing() -> None:
    """The default (animation off) emits no extra message and no frames."""

    async def _run() -> List[Dict[str, Any]]:
        session = _session(build_module(_SPEC), _spikes(), False)
        ws = FakeWS()
        frames = await animation.prepare(ws, session, False)
        assert frames == []
        return ws.sent

    assert asyncio.run(_run()) == []


def test_prepare_reports_unavailable_without_a_model() -> None:
    """Requesting animation with no model reports a named unavailability."""

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        session.set_engine(_Sample(_spikes()), EncodeConfig())
        ws = FakeWS()
        frames = await animation.prepare(ws, session, True)
        assert frames == []
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == ["animation_state"]
    assert sent[0]["payload"]["available"] is False
    assert sent[0]["payload"]["reason"] == animation.NO_MODEL


def test_stream_frames_emits_a_hidden_frame_per_step() -> None:
    """With animate_hidden the run streams input and hidden frames."""
    sent = asyncio.run(_stream(animate=True))
    assert sent[0]["type"] == "animation_state"
    assert sent[0]["payload"]["available"] is True
    frames = [m for m in sent if m["type"] == "spike_frame"]
    hidden = [m for m in frames if m["source"] == "hidden"]
    inputs = [m for m in frames if m["source"] == "input"]
    assert hidden and len(hidden) == len(inputs)
    assert len(hidden[0]["payload"]) == 1
    assert hidden[0]["step"] == 0


def test_stream_frames_keeps_the_input_stream_when_off() -> None:
    """With animate_hidden unset only the input frames stream, as today."""
    sent = asyncio.run(_stream(animate=False))
    assert [m["type"] for m in sent] == ["spike_frame"] * len(sent)
    assert {m["source"] for m in sent} == {"input"}


async def _stream(animate: bool) -> List[Dict[str, Any]]:
    """Return the messages one short run of ``stream_frames`` produced."""
    module = build_module(_SPEC)
    session = _session(module, _spikes(), animate)
    cfg = session.encode_config
    assert cfg is not None
    cfg = cfg.model_copy(update={"interval_ms": _INTERVAL_MS})
    ws = FakeWS()
    task = asyncio.create_task(stream_frames(ws, session, cfg))
    await asyncio.sleep(0.15)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task
    return ws.sent
