"""The ``energy_report`` WebSocket action and its payload shape."""

import asyncio
from typing import Any, Dict, List

from server.handlers import dispatch
from server.schemas import ClientMessage, ServerMessage, TrainConfig
from server.session import Session

_SMALL = TrainConfig(topology="fc_small", hidden=8, num_classes=4)


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


async def _dispatch(message: ClientMessage) -> List[Dict[str, Any]]:
    """Dispatch one message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


def test_energy_report_defaults_to_reference() -> None:
    """A bare action reports the reference target's estimate."""
    sent = asyncio.run(
        _dispatch(ClientMessage(type="energy_report", train=_SMALL))
    )
    assert [message["type"] for message in sent] == ["energy_report"]
    payload = sent[0]["payload"]
    assert payload["target"] == "reference"
    assert payload["report"]["basis"] == "declared cost table"
    assert payload["report"]["estimate"] is True
    assert payload["comparison"]["within_tolerance"] is True


def test_energy_report_named_target() -> None:
    """A named target routes through to its declared cost table."""
    message = ClientMessage(
        type="energy_report", name="norse", train=_SMALL
    )
    sent = asyncio.run(_dispatch(message))
    assert sent[0]["payload"]["target"] == "norse"
    assert sent[0]["payload"]["report"]["energy"] is not None


def test_energy_report_unknown_target_errors() -> None:
    """An unknown target emits the existing error channel."""
    sent = asyncio.run(
        _dispatch(ClientMessage(type="energy_report", name="bogus"))
    )
    assert sent[0]["type"] == "error"
    assert "unknown target" in sent[0]["payload"]


def test_energy_sparse_flag_defaults_on() -> None:
    """The additive request flag defaults to the event-driven path."""
    assert ClientMessage(type="energy_report").sparse is True
    assert ClientMessage(type="energy_report", sparse=False).sparse is False


def test_server_message_accepts_energy_report() -> None:
    """The outbound schema admits the new reply type."""
    message = ServerMessage(type="energy_report", payload={"ok": True})
    assert message.type == "energy_report"
