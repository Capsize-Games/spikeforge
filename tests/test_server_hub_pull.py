"""Dispatch routing and payload shapes for hub_pull and bundle_build."""

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pytest
import torch
from _fakes import FakeWS

import server.hub_payloads as hub_payloads
from server.handlers import dispatch
from server.protocol_handlers import PROTOCOL_ACTIONS
from server.schemas import ClientMessage
from server.session import Session
from spikeforge.network import model_store
from spikeforge.training.training_engine import TrainingEngine
from spikeforge_hub import cache

_COMMUNITY_ENTRY: Dict[str, Any] = {
    "id": "@ada/mnist-fc",
    "name": "mnist-fc",
    "framework": "spikeforge",
    "kind": "bundle",
    "source": "community",
    "license": "BSD-3-Clause",
    "notes": "Community upload, version 1. Checks: unchecked.",
    "url": "https://cdn.spikeforge.net/0123abcd",
    "size_bytes": 7,
}

_NAME = "hub_pull_bundle_ckpt"




async def _dispatch(
    message: ClientMessage, session: Optional[Session] = None
) -> List[Dict[str, Any]]:
    """Dispatch one client message against an in-memory session."""
    owned = session or Session(asyncio.get_running_loop())
    ws = FakeWS()
    await dispatch(ws, owned, message)
    return ws.sent


@pytest.fixture
def sandbox(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Route the hub cache, model store, and bundles into tmp."""
    monkeypatch.setattr(cache, "HUB_CACHE_DIR", str(tmp_path / "hub"))
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr(hub_payloads, "DATA_DIR", str(tmp_path / "data"))
    return tmp_path


def _seed_community(root: Path) -> None:
    """Write one community entry as the cached hosted index."""
    dest = root / "hub" / ".community"
    dest.mkdir(parents=True)
    payload = {"version": 1, "entries": [_COMMUNITY_ENTRY]}
    (dest / "index.json").write_text(json.dumps(payload), encoding="utf-8")


class _FakePullManager:
    """A stand-in pull manager emitting one done snapshot."""

    def __init__(self) -> None:
        """Start with no recorded entry."""
        self.entry_id = ""

    async def ensure(
        self, entry_id: str, emit: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Record the entry and emit a done snapshot."""
        self.entry_id = entry_id
        await emit({
            "id": entry_id,
            "status": "done",
            "bytes": 7,
            "total_bytes": 7,
            "verified": False,
        })


def _fake_index(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the hosted-index refresh with a no-op."""
    monkeypatch.setattr(
        "spikeforge_hub.community.fetch_index", lambda: None
    )


async def _dispatch_pull(
    message: ClientMessage, fake: _FakePullManager
) -> List[Dict[str, Any]]:
    """Dispatch against a session owned by the fake pull manager."""
    session = Session(asyncio.get_running_loop())
    session.hub_pulls = fake
    return await _dispatch(message, session)


def _save_checkpoint() -> None:
    """Train one real step and save a small fc_small checkpoint."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist",
        num_steps=4,
        device="cpu",
        topology="fc_small",
        topology_params={"hidden": 5, "num_classes": 3},
    )
    engine.save(_NAME)


def test_protocol_actions_include_pull_and_bundle_build() -> None:
    """The shared router knows the community pull and bundle actions."""
    assert {"hub_pull", "bundle_build"} <= PROTOCOL_ACTIONS


def test_hub_pull_streams_pull_state(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A community id pulls with hub_pull_state progress snapshots."""
    _seed_community(sandbox)
    _fake_index(monkeypatch)
    fake = _FakePullManager()
    message = ClientMessage(type="hub_pull", name="@ada/mnist-fc")
    sent = asyncio.run(_dispatch_pull(message, fake))
    assert fake.entry_id == "@ada/mnist-fc"
    assert [item["type"] for item in sent] == ["hub_pull_state"]
    payload = sent[0]["payload"]
    assert payload["status"] == "done"
    assert json.dumps(payload)


def test_hub_pull_curated_id_reports_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A curated id sent to hub_pull is redirected, not pulled."""
    _fake_index(monkeypatch)
    fake = _FakePullManager()
    message = ClientMessage(type="hub_pull", name="nir/fc_small")
    sent = asyncio.run(_dispatch_pull(message, fake))
    assert [item["type"] for item in sent] == ["error"]
    assert "use hub_download" in sent[0]["payload"]


def test_hub_pull_missing_name_reports_error() -> None:
    """hub_pull without a model name fails before any refresh."""
    sent = asyncio.run(_dispatch(ClientMessage(type="hub_pull")))
    assert [item["type"] for item in sent] == ["error"]
    assert "requires a model name" in sent[0]["payload"]


def test_bundle_build_reports_identity(sandbox: Path) -> None:
    """A saved model builds to a .spkf whose identity is reported."""
    _save_checkpoint()
    sent = asyncio.run(
        _dispatch(ClientMessage(type="bundle_build", name=_NAME))
    )
    assert [item["type"] for item in sent] == ["bundle_built"]
    payload = sent[0]["payload"]
    assert payload["name"] == _NAME
    assert payload["path"].endswith(f"{_NAME}.spkf")
    assert payload["size_bytes"] > 0
    digest = hashlib.sha256(Path(payload["path"]).read_bytes())
    assert payload["sha256"] == digest.hexdigest()
    assert json.dumps(payload)


def test_bundle_build_unknown_model_reports_error(sandbox: Path) -> None:
    """An unsaved name becomes an error payload, not a raise."""
    message = ClientMessage(type="bundle_build", name="nope")
    sent = asyncio.run(_dispatch(message))
    assert [item["type"] for item in sent] == ["error"]
    assert "no saved model" in sent[0]["payload"]


def test_bundle_build_missing_name_reports_error() -> None:
    """bundle_build without a model name fails before any build."""
    sent = asyncio.run(_dispatch(ClientMessage(type="bundle_build")))
    assert [item["type"] for item in sent] == ["error"]
    assert "requires a model name" in sent[0]["payload"]
