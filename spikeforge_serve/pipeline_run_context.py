"""Shared types for a pipeline run: callbacks and the fixed per-run inputs."""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from spikeforge_serve.pipeline import PipelineGraph

#: Returns a bundle source (path or DeploymentBundle) for one checkpoint name.
CheckpointLoader = Callable[[str], Any]
#: Called after each node finishes, with its id and its JSON-shaped result.
NodeCallback = Callable[[str, Dict[str, Any]], None]
#: Polled between nodes; returning True halts the run before the next one.
StopCheck = Callable[[], bool]


@dataclass(frozen=True)
class RunContext:
    """Fixed inputs shared by every node in one pipeline run."""

    graph: PipelineGraph
    checkpoint_loader: CheckpointLoader
    request: Dict[str, Any]
    device: str
