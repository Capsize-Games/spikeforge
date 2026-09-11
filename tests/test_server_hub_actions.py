"""Dispatch routing and payload shapes for the hub WS actions."""

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

import server.hub_handlers as hub_handlers
from server.downloads import DownloadManager
from server.handlers import dispatch
from server.protocol_handlers import PROTOCOL_ACTIONS
from server.schemas import ClientMessage, HubQuery
from server.session import Session
from snn_interpreter.hub import cache
from snn_interpreter.network import model_store

pytest.importorskip("nir")

_HUB_ACTIONS = {
    "hub_list",
    "hub_search",
    "hub_download",
    "hub_cancel",
    "hub_inspect",
    "hub_import",
}


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


async def _dispatch(message: ClientMessage) -> List[Dict[str, Any]]:
    """Dispatch one client message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache and model store into a temporary directory."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    return tmp_path


def test_protocol_actions_include_every_hub_action() -> None:
    """The shared router knows every hub client action."""
    assert _HUB_ACTIONS <= PROTOCOL_ACTIONS


def test_dispatch_routes_hub_list() -> None:
    """The list action replies with the annotated catalog."""
    sent = asyncio.run(_dispatch(ClientMessage(type="hub_list")))
    assert [message["type"] for message in sent] == ["hub_list"]
    payload = sent[0]["payload"]
    assert len(payload["entries"]) >= 10
    assert payload["issues"] == []
    assert json.dumps(payload)


def test_hub_list_filters_by_framework() -> None:
    """A framework query keeps only that framework's entries."""
    message = ClientMessage(type="hub_list", hub=HubQuery(framework="nir"))
    sent = asyncio.run(_dispatch(message))
    entries = sent[0]["payload"]["entries"]
    assert entries
    assert all(item["framework"] == "nir" for item in entries)


def test_dispatch_routes_hub_search() -> None:
    """The search action reports live availability and results."""
    message = ClientMessage(type="hub_search", hub=HubQuery(query="conv"))
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["hub_search"]
    payload = sent[0]["payload"]
    assert payload["query"] == "conv"
    assert isinstance(payload["available"], bool)
    assert payload["results"]
    assert json.dumps(payload)


def test_hub_inspect_reports_structure() -> None:
    """The inspect action replies with the artifact's structural report."""
    message = ClientMessage(type="hub_inspect", name="nir/fc_small")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["hub_inspect"]
    assert sent[0]["payload"]["kind"] == "nir_graph"


def test_hub_inspect_unknown_reports_error() -> None:
    """An unknown id becomes an error payload, not a raise."""
    message = ClientMessage(type="hub_inspect", name="nope")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["error"]
    assert "unknown catalog entry" in sent[0]["payload"]


def test_hub_import_promotes_compatible(sandbox: Path) -> None:
    """A compatible artifact is promoted and carries its verdict."""
    message = ClientMessage(type="hub_import", name="nir/fc_small")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["hub_import"]
    payload = sent[0]["payload"]
    assert payload["verdict"]["verdict"] == "exact"
    assert payload["promoted"] is True
    assert json.dumps(payload)


def test_hub_download_unknown_reports_error() -> None:
    """Downloading an unknown id fails before any network or child process."""
    message = ClientMessage(type="hub_download", name="nope")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["error"]
    assert "unknown catalog entry" in sent[0]["payload"]


def test_hub_cancel_emits_terminal_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancelling replies with the manager's terminal download state."""

    class FakeManager:
        """A stand-in manager reporting a cancelled snapshot."""

        def cancel(self) -> None:
            """Pretend to terminate an in-flight download."""

        def snapshot(self) -> Dict[str, Any]:
            """Return a terminal cancelled state."""
            return {
                "id": "nir/fc_small",
                "status": "cancelled",
                "bytes": 0,
                "total_bytes": None,
                "verified": False,
            }

    monkeypatch.setattr(hub_handlers, "manager", FakeManager())
    message = ClientMessage(type="hub_cancel", name="nir/fc_small")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["hub_download_state"]
    assert sent[0]["payload"]["status"] == "cancelled"


def test_dataset_download_state_payload_is_unchanged() -> None:
    """The dataset download_state keys stay exactly as they were."""
    assert DownloadManager().snapshot() == {
        "dataset": "",
        "status": "idle",
        "bytes": 0,
    }
