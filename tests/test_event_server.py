"""Server modality routing, event payloads, and clean degradation."""

import asyncio
from typing import Any, Dict, List, Optional

from server import engine_factory
from server.encoder import EncoderEngine
from server.event_engine import EventEngine
from server.handlers import dispatch
from server.introspection_handlers import EVENT_UNSUPPORTED
from server.messages import send_initial
from server.schemas import ClientMessage, EncodeConfig
from server.session import Session
from spikeforge.events import event_source
from spikeforge.events.event_source import EventSampleSource


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


class FakeImageEngine:
    """A stand-in image engine proving the image branch is unchanged."""

    modality = "image"

    def sample_image(self) -> Any:
        """Return a 1x1 sample image."""
        return [[1.0]]

    def reconstruction(self) -> Optional[Dict[str, Any]]:
        """Return a rate reconstruction dict."""
        return {"gain1": [[0.5]], "low": [[0.0]]}

    def raster(self) -> Dict[str, Any]:
        """Return an empty-but-valid raster payload."""
        return {
            "time": [], "neurons": [], "num_steps": 4, "num_neurons": 4,
        }

    def num_steps(self) -> int:
        """Return the step count."""
        return 4

    def target_label(self) -> int:
        """Return the target label."""
        return 0

    def dataset(self) -> str:
        """Return the dataset key."""
        return "mnist"

    def sample_index(self) -> int:
        """Return the sample index."""
        return 0

    def sample_label(self) -> int:
        """Return the sample label."""
        return 0


def _image_session() -> Session:
    """Return a session holding the fake image engine."""
    session = Session(asyncio.get_running_loop())
    session.set_engine(FakeImageEngine(), EncodeConfig(dataset="mnist"))
    return session


def _event_session() -> Session:
    """Return a session holding an offline event engine."""
    session = Session(asyncio.get_running_loop())
    cfg = EncodeConfig(dataset="n_mnist")
    source = EventSampleSource("n_mnist", synthetic_only=True)
    session.set_engine(EventEngine(cfg, source=source), cfg)
    return session


def _build(monkeypatch: Any, dataset: str) -> Any:
    """Run build_encoder against a stubbed download step, offline."""
    async def _ready(ws: Any, session: Any, name: str,
                     train: bool = True) -> bool:
        return True

    monkeypatch.setattr(engine_factory, "ensure_dataset", _ready)

    async def _run() -> Any:
        session = Session(asyncio.get_running_loop())
        ok = await engine_factory.build_encoder(
            FakeWS(), session, EncodeConfig(dataset=dataset)
        )
        return ok, session

    return asyncio.run(_run())


def test_engine_class_routes_by_modality() -> None:
    """Event datasets select the event engine; images keep the old one."""
    assert engine_factory._engine_class("n_mnist") is EventEngine
    assert engine_factory._engine_class("mnist") is EncoderEngine


def test_missing_loader_skips_the_download(
    monkeypatch: Any,
) -> None:
    """An unavailable event loader needs no download; images still do."""
    monkeypatch.setattr(
        engine_factory, "dataset_available", lambda name: False
    )
    assert engine_factory._needs_download("n_mnist") is False
    assert engine_factory._needs_download("mnist") is True


def test_build_encoder_routes_event_dataset(
    monkeypatch: Any,
) -> None:
    """The event dataset builds the event engine, offline and synthetic."""
    monkeypatch.setattr(
        event_source, "dataset_available", lambda name: False
    )
    ok, session = _build(monkeypatch, "n_mnist")
    assert ok is True
    assert isinstance(session.engine, EventEngine)
    assert session.engine.modality == "event"


def test_build_encoder_keeps_the_image_engine(
    monkeypatch: Any,
) -> None:
    """The image dataset still builds the image engine."""
    captured: Dict[str, Any] = {}

    class FakeImage:
        def __init__(self, cfg: EncodeConfig) -> None:
            captured["dataset"] = cfg.dataset

    monkeypatch.setattr(engine_factory, "EncoderEngine", FakeImage)
    ok, session = _build(monkeypatch, "mnist")
    assert ok is True
    assert captured["dataset"] == "mnist"
    assert isinstance(session.engine, FakeImage)


def test_event_send_initial_emits_frame_and_raster_only() -> None:
    """Event init sends a modality ack, one event frame, and the raster."""
    async def _run() -> List[Dict[str, Any]]:
        session = _event_session()
        ws = FakeWS()
        await send_initial(ws, session, session.encode_config)
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == [
        "config_ack", "image", "raster", "status",
    ]
    assert sent[0]["payload"]["modality"] == "event"
    assert sent[1]["kind"] == "event_frame"
    assert sent[2]["source"] == "input"
    assert sent[3]["payload"]["modality"] == "event"
    assert sent[3]["payload"]["num_steps"] == 10


def test_event_config_ack_keeps_every_config_key() -> None:
    """The config_ack stays additive: every config key survives."""
    async def _run() -> Dict[str, Any]:
        session = _event_session()
        ws = FakeWS()
        await send_initial(ws, session, session.encode_config)
        return ws.sent[0]["payload"]

    ack = asyncio.run(_run())
    for key in EncodeConfig().model_dump():
        assert key in ack


def test_event_init_sends_no_reconstruction() -> None:
    """Event modality never emits a rate reconstruction."""
    async def _run() -> List[Dict[str, Any]]:
        session = _event_session()
        ws = FakeWS()
        await send_initial(ws, session, session.encode_config)
        return ws.sent

    kinds = {
        m.get("kind") for m in asyncio.run(_run()) if m["type"] == "image"
    }
    assert kinds == {"event_frame"}


def test_image_init_keeps_sample_and_reconstruction() -> None:
    """The image path still sends the sample and both reconstructions."""
    async def _run() -> List[Dict[str, Any]]:
        session = _image_session()
        ws = FakeWS()
        await send_initial(ws, session, session.encode_config)
        return ws.sent

    sent = asyncio.run(_run())
    kinds = {m.get("kind") for m in sent if m["type"] == "image"}
    assert sent[0]["payload"]["modality"] == "image"
    assert kinds == {None, "recon_gain1", "recon_low"}


def test_event_encoding_report_degrades_cleanly() -> None:
    """An image-only action returns a typed error for event modality."""
    async def _run() -> List[Dict[str, Any]]:
        ws = FakeWS()
        await dispatch(
            ws, _event_session(), ClientMessage(type="encoding_report")
        )
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == EVENT_UNSUPPORTED


def test_event_infer_without_model_errors_not_raises() -> None:
    """Inference without a model reports an error instead of raising."""
    async def _run() -> List[Dict[str, Any]]:
        ws = FakeWS()
        await dispatch(ws, _event_session(), ClientMessage(type="infer"))
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == "no model loaded; train or load one"
