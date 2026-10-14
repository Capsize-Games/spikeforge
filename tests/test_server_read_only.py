"""Read-only demo matrix: every client action with writes disabled.

With ``SPIKEFORGE_DASHBOARD_READ_ONLY=1`` the write actions (train,
model save/load/delete/new, hub download/inspect/import/pull, pipeline
save/delete/run) are refused with a read-only error, while every other
action routes normally. With the flag unset the gates stay open.
"""

import asyncio
import contextlib
from typing import Any, Dict, List, get_args

import pytest
from _fakes import FakeWS

import server.handlers as handlers
from server.demo import READ_ONLY_ENV
from server.handlers import dispatch
from server.schemas import ClientMessage
from server.schemas.hub_query import HubQuery
from server.session import Session

#: Actions refused while the demo is read-only.
REFUSED = frozenset({
    "train",
    "save_model", "load_model", "delete_model", "new_model",
    "hub_download", "hub_inspect", "hub_import", "hub_pull",
    "save_pipeline", "delete_pipeline", "run_pipeline",
})

_REFUSAL = "disabled on the public demo"




def _all_types() -> List[str]:
    """Return every ClientMessage type the protocol declares."""
    field = ClientMessage.model_fields["type"]
    return sorted(get_args(field.annotation))


async def _build_noop(*args: Any, **kwargs: Any) -> bool:
    """Stand in for engine builds, which need real datasets."""
    return False


async def _dispatch(
    message: ClientMessage, monkeypatch: pytest.MonkeyPatch
) -> List[Dict[str, Any]]:
    """Dispatch one message with dataset-backed builds stubbed out.

    A raised handler error counts as routed, not refused: the socket
    loop turns it into an ``error`` frame instead of propagating it.
    """
    monkeypatch.setattr(handlers, "build_encoder", _build_noop)
    session = Session(asyncio.get_running_loop())
    ws = FakeWS()
    with contextlib.suppress(Exception):
        await dispatch(ws, session, message)
    return ws.sent


def _refused(sent: List[Dict[str, Any]]) -> bool:
    """Return True when a read-only refusal was sent."""
    for item in sent:
        payload = item.get("payload")
        if (
            item.get("type") == "error"
            and isinstance(payload, str)
            and _REFUSAL in payload
        ):
            return True
    return False


def _message(action: str) -> ClientMessage:
    """Return a minimal message exercising ``action``'s gate."""
    raw: Dict[str, Any] = {"type": action}
    if action == "deploy_run":
        raw["name"] = "no-such-target"
    elif action == "hub_search":
        raw["hub"] = HubQuery(query="fc").model_dump()
    elif action == "load_pipeline":
        raw["name"] = "no-such-pipeline"
    return ClientMessage.model_validate(raw)


@pytest.fixture
def read_only(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Enable demo read-only mode for one test."""
    monkeypatch.setenv(READ_ONLY_ENV, "1")
    return monkeypatch


def test_matrix_covers_every_client_action() -> None:
    """The refused/open split accounts for all protocol actions."""
    declared = set(_all_types())
    assert declared >= REFUSED
    assert len(declared) > len(REFUSED)


@pytest.mark.parametrize("action", sorted(REFUSED))
def test_write_actions_refused(
    read_only: pytest.MonkeyPatch, action: str
) -> None:
    """Write actions are refused while the demo is read-only."""
    sent = asyncio.run(_dispatch(_message(action), read_only))
    assert _refused(sent), f"{action} was not refused: {sent!r}"


@pytest.mark.parametrize(
    "action", sorted(set(_all_types()) - REFUSED)
)
def test_read_actions_not_refused(
    read_only: pytest.MonkeyPatch, action: str
) -> None:
    """Read actions still route while the demo is read-only."""
    sent = asyncio.run(_dispatch(_message(action), read_only))
    assert not _refused(sent), f"{action} was refused: {sent!r}"


@pytest.mark.parametrize(
    "action", ["hub_download", "save_pipeline", "delete_pipeline"]
)
def test_gates_open_with_flag_unset(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """The same actions route past the gate with the flag unset."""
    monkeypatch.delenv(READ_ONLY_ENV, raising=False)
    sent = asyncio.run(_dispatch(_message(action), monkeypatch))
    assert not _refused(sent)
    assert sent and sent[0]["type"] == "error"
    assert "requires a" in sent[0]["payload"]
