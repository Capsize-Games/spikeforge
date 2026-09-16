"""Execute a pipeline graph: one ServingService call per node, in order.

Deliberately symmetric with :mod:`spikeforge_serve.module_runner`'s
one-shot path -- a pipeline node runs through the exact same
:class:`~spikeforge_serve.service.ServingService` a standalone module does,
just with its input built from an upstream node's output instead of a
request file.
"""

from typing import Any, Dict, List, Optional, Tuple

from spikeforge_serve.payloads import (
    encoded_flag,
    frames_from,
    prediction_json,
)
from spikeforge_serve.pipeline import PipelineEdge, PipelineGraph, PipelineNode
from spikeforge_serve.pipeline_run_context import (
    CheckpointLoader,
    NodeCallback,
    RunContext,
    StopCheck,
)
from spikeforge_serve.service import ServingService


class PipelineRunError(RuntimeError):
    """A node failed to build or run; the message names which one."""


def _extract(
    edge: PipelineEdge,
    upstream_result: Dict[str, Any],
    target_num_classes: Optional[int],
) -> List[float]:
    """Shape an upstream node's prediction into the target's next frame."""
    if edge.extract == "mean_logits":
        return list(upstream_result["mean_logits"]["values"][0])
    if edge.extract == "predicted_class":
        return [float(upstream_result["predicted"])]
    if edge.extract == "one_hot":
        index = int(upstream_result["predicted"])
        size = target_num_classes if target_num_classes else index + 1
        vector = [0.0] * size
        vector[index] = 1.0
        return vector
    raise PipelineRunError(  # pragma: no cover - guarded by PipelineEdge
        f"edge {edge.id!r}: unknown extract mode {edge.extract!r}"
    )


def run_pipeline(
    graph: PipelineGraph,
    request: Dict[str, Any],
    checkpoint_loader: CheckpointLoader,
    on_node_done: Optional[NodeCallback] = None,
    device: str = "cpu",
    should_stop: Optional[StopCheck] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run every node in topological order; return ``{node_id: result}``.

    Fan-in (a node with more than one incoming edge) is refused -- a
    deliberate v1 scope cut (see documentation/model-deployment.md).
    """
    ctx = RunContext(graph, checkpoint_loader, request, device)
    results: Dict[str, Dict[str, Any]] = {}
    for node in graph.topological_order():
        if should_stop is not None and should_stop():
            break
        _process_node(ctx, node, results, on_node_done)
    return results


def _process_node(
    ctx: RunContext,
    node: PipelineNode,
    results: Dict[str, Dict[str, Any]],
    on_node_done: Optional[NodeCallback],
) -> None:
    """Run one node, record its result, and fire the callback."""
    payload = _run_node(ctx, node, results)
    results[node.id] = payload
    if on_node_done is not None:
        on_node_done(node.id, payload)


def _run_node(
    ctx: RunContext, node: PipelineNode, results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """Load the node's checkpoint, build its input, and predict."""
    incoming = ctx.graph.incoming(node.id)
    if len(incoming) > 1:
        raise PipelineRunError(
            f"node {node.id!r}: fan-in is not supported"
        )
    try:
        service = ServingService(
            ctx.checkpoint_loader(node.checkpoint), device=ctx.device
        )
        frames, encoded = _node_input(ctx, node, incoming, results, service)
        predictions = service.predict(frames, encoded=encoded)
        return prediction_json(predictions[-1])
    except Exception as exc:
        raise PipelineRunError(
            f"node {node.id!r} ({node.checkpoint!r}): {exc}"
        ) from exc


def _node_input(
    ctx: RunContext,
    node: PipelineNode,
    incoming: List[PipelineEdge],
    results: Dict[str, Dict[str, Any]],
    service: ServingService,
) -> Tuple[List[Any], bool]:
    """Return the ``(frames, encoded)`` input for one node.

    The first element is a list of *frames*, matching what ``frames_from``
    yields and what ``ServingService.predict`` consumes. An upstream node
    contributes exactly one frame, so its vector is wrapped rather than
    spread.
    """
    if not incoming:
        return frames_from(ctx.request), encoded_flag(ctx.request)
    edge = incoming[0]
    num_classes = service.bundle.manifest.get("num_classes")
    frame = _extract(edge, results[edge.source], num_classes)
    return [frame], True
