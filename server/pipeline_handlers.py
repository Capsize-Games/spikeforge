"""Handlers for the pipeline CRUD, run, and stop WebSocket actions.

Mirrors :mod:`server.model_handlers`'s shape: list/save/load/delete are
synchronous file I/O handled directly here, while run/stop delegate to
``session.pipeline`` (a :class:`~server.pipeline_service.PipelineService`),
the same background-thread pattern :mod:`server.training` uses.
"""

from typing import Optional

from fastapi import WebSocket

from server.concurrency import SERVER_BUSY_MESSAGE
from server.messages import send_locked
from server.schemas import ClientMessage
from server.session import Session
from spikeforge_serve import pipeline_store
from spikeforge_serve.pipeline import PipelineGraph, PipelineGraphError

PIPELINE_ACTIONS = frozenset({
    "list_pipelines", "save_pipeline", "load_pipeline",
    "delete_pipeline", "run_pipeline", "stop_pipeline",
})


async def handle_list_pipelines(
    ws: WebSocket, session: Session, _message: ClientMessage
) -> None:
    """Emit every saved pipeline's summary metadata."""
    await send_locked(ws, session, {
        "type": "pipeline_list",
        "payload": {"pipelines": pipeline_store.list_pipelines()},
    })


async def _parsed_graph(
    ws: WebSocket, session: Session, message: ClientMessage
) -> Optional[PipelineGraph]:
    """Parse and validate the pipeline graph, reporting a typed error."""
    try:
        graph = PipelineGraph.from_dict(message.pipeline.model_dump())
        graph.topological_order()  # also catches a cycle before saving
        return graph
    except PipelineGraphError as exc:
        await send_locked(ws, session, {"type": "error", "payload": str(exc)})
        return None


async def handle_save_pipeline(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Validate and persist the current graph under the given name."""
    name = message.name or message.pipeline.name
    if not name:
        await send_locked(ws, session, {
            "type": "error", "payload": "save_pipeline requires a name",
        })
        return
    graph = await _parsed_graph(ws, session, message)
    if graph is None:
        return
    graph_dict = dict(graph.to_dict())
    graph_dict["name"] = name
    pipeline_store.save(name, graph_dict)
    await send_locked(ws, session, {
        "type": "pipeline_saved", "payload": {"name": name},
    })


async def handle_load_pipeline(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Emit one saved pipeline's full graph."""
    if not message.name:
        await send_locked(ws, session, {
            "type": "error", "payload": "load_pipeline requires a name",
        })
        return
    try:
        graph_dict = pipeline_store.load(message.name)
    except FileNotFoundError as exc:
        await send_locked(ws, session, {"type": "error", "payload": str(exc)})
        return
    await send_locked(ws, session, {
        "type": "pipeline_loaded", "payload": graph_dict,
    })


async def handle_delete_pipeline(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Remove one saved pipeline; report whether it existed."""
    if not message.name:
        await send_locked(ws, session, {
            "type": "error", "payload": "delete_pipeline requires a name",
        })
        return
    existed = pipeline_store.delete(message.name)
    await send_locked(ws, session, {
        "type": "pipeline_deleted",
        "payload": {"name": message.name, "existed": existed},
    })


async def handle_run_pipeline(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Start the pipeline worker unless one is already running."""
    if session.pipeline.is_running:
        return
    if not session.pipeline.start(message.pipeline, message.pipeline_input):
        await send_locked(ws, session, {
            "type": "error", "payload": SERVER_BUSY_MESSAGE,
        })


async def handle_stop_pipeline(
    _ws: WebSocket, session: Session, _message: ClientMessage
) -> None:
    """Signal the running pipeline worker to stop after its current node."""
    session.pipeline.stop()


async def dispatch_pipeline(
    ws: WebSocket, session: Session, message: ClientMessage
) -> None:
    """Route one pipeline CRUD, run, or stop action."""
    if message.type == "list_pipelines":
        await handle_list_pipelines(ws, session, message)
    elif message.type == "save_pipeline":
        await handle_save_pipeline(ws, session, message)
    elif message.type == "load_pipeline":
        await handle_load_pipeline(ws, session, message)
    elif message.type == "delete_pipeline":
        await handle_delete_pipeline(ws, session, message)
    elif message.type == "run_pipeline":
        await handle_run_pipeline(ws, session, message)
    else:
        await handle_stop_pipeline(ws, session, message)
