"""Dispatch routing and payload shapes for the pipeline WebSocket actions."""

import asyncio
from typing import Any, Dict, List

import pytest
import torch

from server.handlers import dispatch
from server.schemas import (
    ClientMessage,
    PipelineEdgeConfig,
    PipelineGraphConfig,
    PipelineNodeConfig,
)
from server.session import Session
from spikeforge.network import model_store
from spikeforge.training.training_engine import TrainingEngine


@pytest.fixture(autouse=True)
def _model_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect checkpoint reads/writes into a per-test directory."""
    monkeypatch.setattr(model_store, "MODEL_DIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _pipelines_dir(tmp_path: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Redirect pipeline reads/writes into a per-test directory."""
    from spikeforge_serve import pipeline_store

    monkeypatch.setattr(pipeline_store, "PIPELINES_DIR", str(tmp_path / "p"))


def _save_checkpoint(name: str, num_classes: int) -> None:
    """Save a tiny, deterministic fc_small checkpoint (784-feature input)."""
    torch.manual_seed(0)
    engine = TrainingEngine(
        dataset="mnist", num_steps=3, device="cpu", topology="fc_small",
        topology_params={"hidden": 4, "num_classes": num_classes},
    )
    engine.save(name)


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


async def _dispatch(message: ClientMessage) -> "tuple[FakeWS, Session]":
    """Dispatch one client message against a fresh in-memory session."""
    session = Session(asyncio.get_running_loop())
    ws = FakeWS()
    await dispatch(ws, session, message)
    return ws, session


async def _drain_pipeline(session: Session) -> List[Dict[str, Any]]:
    """Collect every queued pipeline message until the run reports done."""
    messages: List[Dict[str, Any]] = []
    while True:
        message = await asyncio.wait_for(session.pipeline.queue.get(), 10)
        messages.append(message)
        payload = message.get("payload")
        if (
            message["type"] == "pipeline_run_state"
            and isinstance(payload, dict)
            and payload.get("running") is False
        ):
            return messages


def _one_node_graph(checkpoint: str) -> PipelineGraphConfig:
    return PipelineGraphConfig(
        name="g", nodes=[PipelineNodeConfig(id="a", checkpoint=checkpoint)],
    )


def _raw_image() -> List[List[List[float]]]:
    return [[[0.0] * 28 for _ in range(28)]]


def test_list_pipelines_starts_empty() -> None:
    """No saved pipelines is an empty list, not an error."""
    ws, _session = asyncio.run(_dispatch(ClientMessage(type="list_pipelines")))
    assert [item["type"] for item in ws.sent] == ["pipeline_list"]
    assert ws.sent[0]["payload"]["pipelines"] == []


def test_save_then_list_then_load_round_trips() -> None:
    """A saved graph shows up in the list and loads back intact."""
    graph = PipelineGraphConfig(
        name="ignored",
        nodes=[PipelineNodeConfig(id="a", checkpoint="digits")],
        edges=[],
    )
    ws, _session = asyncio.run(_dispatch(
        ClientMessage(type="save_pipeline", name="my-pipeline", pipeline=graph)
    ))
    assert [item["type"] for item in ws.sent] == ["pipeline_saved"]
    assert ws.sent[0]["payload"]["name"] == "my-pipeline"

    ws, _session = asyncio.run(_dispatch(ClientMessage(type="list_pipelines")))
    assert [p["name"] for p in ws.sent[0]["payload"]["pipelines"]] == [
        "my-pipeline",
    ]

    ws, _session = asyncio.run(
        _dispatch(ClientMessage(type="load_pipeline", name="my-pipeline"))
    )
    assert [item["type"] for item in ws.sent] == ["pipeline_loaded"]
    assert ws.sent[0]["payload"]["nodes"][0]["checkpoint"] == "digits"


def test_save_pipeline_requires_a_name() -> None:
    """Saving with neither a name nor a graph name is a clear error."""
    ws, _session = asyncio.run(_dispatch(
        ClientMessage(type="save_pipeline", pipeline=PipelineGraphConfig())
    ))
    assert [item["type"] for item in ws.sent] == ["error"]
    assert "name" in ws.sent[0]["payload"]


def test_save_pipeline_rejects_a_cycle() -> None:
    """A graph with a cycle is refused before it's ever written to disk."""
    graph = PipelineGraphConfig(
        name="g",
        nodes=[
            PipelineNodeConfig(id="a", checkpoint="x"),
            PipelineNodeConfig(id="b", checkpoint="y"),
        ],
        edges=[
            PipelineEdgeConfig(id="e1", source="a", target="b"),
            PipelineEdgeConfig(id="e2", source="b", target="a"),
        ],
    )
    ws, _session = asyncio.run(_dispatch(
        ClientMessage(type="save_pipeline", name="cyclic", pipeline=graph)
    ))
    assert [item["type"] for item in ws.sent] == ["error"]
    assert "cycle" in ws.sent[0]["payload"]


def test_load_pipeline_requires_a_name() -> None:
    """Loading without a name is a clear error, not a crash."""
    ws, _session = asyncio.run(_dispatch(ClientMessage(type="load_pipeline")))
    assert [item["type"] for item in ws.sent] == ["error"]


def test_load_missing_pipeline_reports_an_error() -> None:
    """Loading a name that was never saved reports it, not a traceback."""
    ws, _session = asyncio.run(
        _dispatch(ClientMessage(type="load_pipeline", name="ghost"))
    )
    assert [item["type"] for item in ws.sent] == ["error"]


def test_delete_pipeline_round_trip() -> None:
    """Deleting a saved pipeline reports existed=True, then False again."""
    graph = _one_node_graph("digits")
    asyncio.run(_dispatch(
        ClientMessage(type="save_pipeline", name="p", pipeline=graph)
    ))
    ws, _session = asyncio.run(
        _dispatch(ClientMessage(type="delete_pipeline", name="p"))
    )
    assert ws.sent[0]["payload"] == {"name": "p", "existed": True}

    ws, _session = asyncio.run(
        _dispatch(ClientMessage(type="delete_pipeline", name="p"))
    )
    assert ws.sent[0]["payload"] == {"name": "p", "existed": False}


def test_run_pipeline_streams_node_result_then_finishes() -> None:
    """Running a real one-node pipeline streams a result and then stops."""
    _save_checkpoint("digits", num_classes=3)
    message = ClientMessage(
        type="run_pipeline",
        pipeline=_one_node_graph("digits"),
        pipeline_input={"frames": [_raw_image()], "encoded": False},
    )

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        ws = FakeWS()
        await dispatch(ws, session, message)
        return await _drain_pipeline(session)

    messages = asyncio.run(_run())
    types = [m["type"] for m in messages]
    assert types[0] == "pipeline_run_state"
    assert messages[0]["payload"]["running"] is True
    assert "pipeline_node_result" in types
    node_result = next(
        m for m in messages if m["type"] == "pipeline_node_result"
    )
    assert node_result["payload"]["node_id"] == "a"
    assert types[-1] == "pipeline_run_state"
    assert messages[-1]["payload"] == {"running": False, "reason": "finished"}


def test_run_pipeline_with_a_missing_checkpoint_reports_an_error() -> None:
    """A node referencing a checkpoint that doesn't exist surfaces cleanly."""
    message = ClientMessage(
        type="run_pipeline",
        pipeline=_one_node_graph("ghost"),
        pipeline_input={"frames": [_raw_image()], "encoded": False},
    )

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        ws = FakeWS()
        await dispatch(ws, session, message)
        return await _drain_pipeline(session)

    messages = asyncio.run(_run())
    types = [m["type"] for m in messages]
    assert "error" in types
    assert messages[-1]["payload"]["reason"] == "error"


def test_stop_pipeline_signals_the_running_worker() -> None:
    """stop_pipeline sets the stop flag without raising when idle too."""

    async def _run() -> Session:
        session = Session(asyncio.get_running_loop())
        ws = FakeWS()
        await dispatch(ws, session, ClientMessage(type="stop_pipeline"))
        return session

    session = asyncio.run(_run())
    assert session.pipeline._stop.is_set()
