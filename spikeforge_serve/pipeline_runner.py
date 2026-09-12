"""Execute a pipeline graph: one ServingService call per node, in order.

Deliberately symmetric with :mod:`spikeforge_serve.module_runner`'s
one-shot path -- a pipeline node runs through the exact same
:class:`~spikeforge_serve.service.ServingService` a standalone module does,
just with its input built from an upstream node's output instead of a
request file.
"""

from typing import Any, Callable, Dict, List, Optional

from spikeforge_serve.payloads import (
    encoded_flag,
    frames_from,
    prediction_json,
)
from spikeforge_serve.pipeline import PipelineEdge, PipelineGraph
from spikeforge_serve.service import ServingService

#: Returns a bundle source (path or DeploymentBundle) for one checkpoint name.
CheckpointLoader = Callable[[str], Any]
#: Called after each node finishes, with its id and its JSON-shaped result.
NodeCallback = Callable[[str, Dict[str, Any]], None]
#: Polled between nodes; returning True halts the run before the next one.
StopCheck = Callable[[], bool]


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

    ``request`` (``{"frames": [...], "encoded": ...}``, the same shape
    :mod:`spikeforge_serve.module_runner` reads from stdin) feeds every node
    with no incoming edge. A node with more than one incoming edge is
    refused -- fan-in merge semantics are an explicit v1 scope cut, not an
    oversight (see documentation/model-deployment.md).
    """
    order = graph.topological_order()
    results: Dict[str, Dict[str, Any]] = {}
    for node in order:
        if should_stop is not None and should_stop():
            break
        incoming = graph.incoming(node.id)
        if len(incoming) > 1:
            raise PipelineRunError(
                f"node {node.id!r}: fan-in is not supported"
            )
        try:
            bundle = checkpoint_loader(node.checkpoint)
            service = ServingService(bundle, device=device)
            if not incoming:
                frames = frames_from(request)
                encoded = encoded_flag(request)
            else:
                edge = incoming[0]
                num_classes = service.bundle.manifest.get("num_classes")
                frame = _extract(edge, results[edge.source], num_classes)
                frames = [frame]
                encoded = True
            predictions = service.predict(frames, encoded=encoded)
            payload = prediction_json(predictions[-1])
        except Exception as exc:
            raise PipelineRunError(
                f"node {node.id!r} ({node.checkpoint!r}): {exc}"
            ) from exc
        results[node.id] = payload
        if on_node_done is not None:
            on_node_done(node.id, payload)
    return results
