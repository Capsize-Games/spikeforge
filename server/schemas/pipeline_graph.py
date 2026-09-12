"""Inbound pipeline graph schema.

Shallow structural validation only -- semantic checks (cycles, unknown
node references, fan-in) live in
:class:`spikeforge_serve.pipeline.PipelineGraph`, which every handler
parses this into before acting on it. Duplicating that logic here as a
second Pydantic-level check would just be two places to keep in sync.
"""

from typing import Dict, List, Literal

from pydantic import BaseModel, Field


class PipelineNodeConfig(BaseModel):
    """One node: a saved checkpoint plus its canvas position."""

    id: str
    checkpoint: str
    position: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0.0, "y": 0.0}
    )


class PipelineEdgeConfig(BaseModel):
    """One edge: the source node's output feeds the target via `extract`."""

    id: str
    source: str
    target: str
    extract: Literal["mean_logits", "predicted_class", "one_hot"] = (
        "mean_logits"
    )


class PipelineGraphConfig(BaseModel):
    """A DAG of checkpoints: nodes plus the edges wiring their outputs."""

    version: int = 1
    name: str = ""
    nodes: List[PipelineNodeConfig] = Field(default_factory=list)
    edges: List[PipelineEdgeConfig] = Field(default_factory=list)
