"""Client-controlled size bounds and inbox backpressure.

Oversized training/encoding configs and hub queries are rejected by
validation before any allocation, and a flooding connection gets its
excess messages dropped instead of growing the server's memory.
"""

import asyncio
import contextlib
from typing import Any, Dict, List

import pytest
from _fakes import FakeWS
from fastapi import WebSocketDisconnect
from pydantic import ValidationError

from server.app import _serve
from server.protocol_version import PROTOCOL_VERSION
from server.schemas import ClientMessage, EncodeConfig, TrainConfig
from server.schemas.hub_query import HubQuery
from server.session import INBOX_MAXSIZE, RATE_BURST, Session


def test_defaults_still_validate_unchanged() -> None:
    """The reference defaults sit inside the new bounds."""
    train = TrainConfig()
    assert (train.hidden, train.epochs, train.num_steps) == (256, 3, 25)
    assert (train.subset, train.batch_size) == (10, 64)
    encode = EncodeConfig()
    assert (encode.num_steps, encode.batch_size) == (100, 128)
    assert HubQuery().limit == 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hidden": 10 ** 9},
        {"hidden": 0},
        {"epochs": 10 ** 6},
        {"num_steps": 10 ** 6},
        {"subset": 10 ** 9},
        {"batch_size": 10 ** 6},
        {"batch_size": 0},
        {"bptt_steps": 10 ** 9},
    ],
)
def test_train_config_bounds_reject_absurd(
    kwargs: Dict[str, int]
) -> None:
    """Absurd training sizes fail validation before any allocation."""
    with pytest.raises(ValidationError):
        TrainConfig.model_validate(kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_steps": 10 ** 6},
        {"batch_size": 10 ** 6},
        {"subset": 0},
        {"sample_index": -1},
        {"input_size": (10 ** 5, 10 ** 5)},
        {"input_size": (0, 28)},
        {"interval_ms": -5},
    ],
)
def test_encode_config_bounds_reject_absurd(
    kwargs: Dict[str, Any]
) -> None:
    """Absurd encoding sizes fail validation before any allocation."""
    with pytest.raises(ValidationError):
        EncodeConfig(**kwargs)


@pytest.mark.parametrize("limit", [-1, 0, 10 ** 6])
def test_hub_query_limit_bounded(limit: int) -> None:
    """A negative or huge hub limit is rejected, not honored."""
    with pytest.raises(ValidationError):
        HubQuery(query="fc", limit=limit)


def test_inbox_full_drops_without_growth() -> None:
    """Past the inbox cap, offers drop and the queue stays bounded."""

    async def scenario() -> None:
        session = Session(asyncio.get_running_loop())
        message = ClientMessage(type="stop")
        for _ in range(INBOX_MAXSIZE):
            assert session.offer(message) is None
        assert session.inbox.qsize() == INBOX_MAXSIZE
        assert session.offer(message) == "full"
        assert session.inbox.qsize() == INBOX_MAXSIZE

    asyncio.run(scenario())


def test_burst_rate_drops_and_recovers() -> None:
    """Past the burst rate offers drop; a drained client recovers."""

    async def scenario() -> None:
        session = Session(asyncio.get_running_loop())
        message = ClientMessage(type="stop")
        for _ in range(RATE_BURST):
            assert session.offer(message) is None
            session.inbox.get_nowait()
        assert session.offer(message) == "limited"
        session._arrivals.clear()
        assert session.offer(message) is None

    asyncio.run(scenario())


def test_socket_flood_reports_drops() -> None:
    """A flooding socket gets drop errors instead of unbounded queue."""

    async def scenario() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        frames = [
            {"protocol_version": PROTOCOL_VERSION, "type": "stop"}
            for _ in range(INBOX_MAXSIZE + 10)
        ]
        ws = FakeWS(frames)
        with contextlib.suppress(WebSocketDisconnect):
            await _serve(ws, session)
        assert session.inbox.qsize() == INBOX_MAXSIZE
        return ws.sent

    sent = asyncio.run(scenario())
    drops = [
        item for item in sent
        if item["type"] == "error"
        and "inbox full" in str(item["payload"])
    ]
    assert len(drops) == 10
