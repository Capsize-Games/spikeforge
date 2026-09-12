"""Run a pipeline graph in a worker thread and stream node results.

Mirrors :class:`server.training.TrainingService`'s bridge exactly: a daemon
thread does the (potentially slow) work and hands results back to the
event loop via ``call_soon_threadsafe``, so the WebSocket read loop is
never blocked on a multi-node run.
"""

import asyncio
import threading
from typing import Any, Dict, Optional

from server import concurrency
from server.schemas import PipelineGraphConfig
from spikeforge.serving import bundle as bundle_mod
from spikeforge_serve.pipeline import PipelineGraph
from spikeforge_serve.pipeline_runner import run_pipeline


class PipelineService:
    """Bridge a blocking pipeline run into an asyncio-friendly queue."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind the service to the loop that owns the send queue."""
        self._loop = loop
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.queue: asyncio.Queue = asyncio.Queue()

    @property
    def is_running(self) -> bool:
        """Return True while the worker thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def start(
        self, graph_config: PipelineGraphConfig, request: Dict[str, Any]
    ) -> bool:
        """Spawn a worker that runs the pipeline and streams node results.

        Returns False without starting anything when the server-wide
        training/pipeline job cap (see ``server.concurrency``) is full.
        """
        if not concurrency.try_acquire():
            return False
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run_job, args=(graph_config, request), daemon=True
        )
        self._thread.start()
        return True

    def stop(self) -> None:
        """Signal the worker to stop after the current node."""
        self._stop.set()

    def _run_job(
        self, graph_config: PipelineGraphConfig, request: Dict[str, Any]
    ) -> None:
        """Run the pipeline, always freeing the job slot afterward."""
        try:
            self._run(graph_config, request)
        finally:
            concurrency.release()

    def _run(
        self, graph_config: PipelineGraphConfig, request: Dict[str, Any]
    ) -> None:
        """Parse the graph, run it, and stream a message per node."""
        self._put({"type": "pipeline_run_state", "payload": {"running": True}})
        try:
            graph = PipelineGraph.from_dict(graph_config.model_dump())
        except Exception as exc:  # a malformed graph never reaches the runner
            self._error(exc)
            return

        def on_node_done(node_id: str, payload: Dict[str, Any]) -> None:
            self._put({
                "type": "pipeline_node_result",
                "payload": {"node_id": node_id, "result": payload},
            })

        try:
            run_pipeline(
                graph,
                request,
                checkpoint_loader=bundle_mod.build,
                on_node_done=on_node_done,
                should_stop=self._stop.is_set,
            )
            reason = "stopped" if self._stop.is_set() else "finished"
            self._put({
                "type": "pipeline_run_state",
                "payload": {"running": False, "reason": reason},
            })
        except Exception as exc:  # surface a node failure to the client
            self._error(exc)
        finally:
            self._thread = None

    def _error(self, exc: Exception) -> None:
        """Queue an error message and a stopped run_state together."""
        self._put({"type": "error", "payload": str(exc)})
        self._put({
            "type": "pipeline_run_state",
            "payload": {"running": False, "reason": "error"},
        })
        self._thread = None

    def _put(self, message: Dict[str, Any]) -> None:
        """Thread-safe enqueue onto the event loop."""
        self._loop.call_soon_threadsafe(self.queue.put_nowait, message)
