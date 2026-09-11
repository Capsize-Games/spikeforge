"""Dispatch routing and payload shapes for the model-registry WS actions."""

import asyncio
import json
from typing import Any, Dict, List, Optional

import pytest
import torch

from server.handlers import dispatch
from server.schemas import ClientMessage, ModelQuery
from server.session import Session
from spikeforge.network import model_store


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint writes into a per-test temporary directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


def _save(
    name: str, dataset: str, coding: str, accuracy: Optional[float]
) -> None:
    """Write one checkpoint with a minimal manifest."""
    meta = {
        "dataset": dataset,
        "topology": "fc_legacy",
        "coding": coding,
        "input_mode": coding,
        "device": "cpu",
    }
    history = (
        [] if accuracy is None else [{"step": 1, "test_accuracy": accuracy}]
    )
    manifest = {"config_hash": "h", "config": meta}
    model_store.save(name, torch.nn.Linear(4, 2), meta, history, manifest)


async def _dispatch(message: ClientMessage) -> List[Dict[str, Any]]:
    """Dispatch one client message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws.sent


def test_dispatch_routes_model_search() -> None:
    """The search action replies with the filtered records."""
    _save("a", "mnist", "rate", 90.0)
    _save("b", "cifar10", "rate", 40.0)
    message = ClientMessage(
        type="model_search", query=ModelQuery(dataset="mnist")
    )
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["model_search"]
    payload = sent[0]["payload"]
    assert [item["name"] for item in payload["models"]] == ["a"]
    assert json.dumps(payload)


def test_dispatch_routes_model_diff() -> None:
    """The diff action replies with the classified metadata union."""
    _save("a", "mnist", "rate", 90.0)
    _save("b", "cifar10", "rate", 40.0)
    message = ClientMessage(
        type="model_diff", query=ModelQuery(names=["a", "b"])
    )
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["model_diff"]
    assert sent[0]["payload"]["changed"] == ["dataset"]
    assert json.dumps(sent[0]["payload"])


def test_model_diff_requires_two_names() -> None:
    """A diff request without a pair of names reports an error."""
    message = ClientMessage(type="model_diff", query=ModelQuery(names=["a"]))
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["error"]
    assert "two names" in sent[0]["payload"]


def test_model_diff_reports_missing_checkpoint() -> None:
    """A diff against a missing checkpoint reports it without raising."""
    _save("a", "mnist", "rate", 90.0)
    message = ClientMessage(
        type="model_diff", query=ModelQuery(names=["a", "ghost"])
    )
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["error"]
    assert "ghost" in sent[0]["payload"]
