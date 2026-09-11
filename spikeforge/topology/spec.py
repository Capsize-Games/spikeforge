"""The declarative topology graph and its graph-builder helpers."""

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence

from spikeforge.topology.edge import Edge
from spikeforge.topology.stage import Stage
from spikeforge.topology.validation import (
    assert_acyclic,
    assert_reachable,
    assert_references,
    assert_stage_exists,
)


@dataclass(frozen=True)
class TopologySpec:
    """A directed graph of stages: the single source of truth.

    ``stages`` are listed in topological order, ``edges`` connect them, and
    ``input``/``output`` name the external entry and readout stages. The
    stage named by ``input`` receives the tensor passed to
    :meth:`~spikeforge.topology.stage_module.StageModule.step`.
    """

    stages: Sequence[Stage]
    edges: Sequence[Edge]
    input: str
    output: str

    def stage(self, name: str) -> Stage:
        """Return the stage called ``name`` or raise ``KeyError``."""
        for stage in self.stages:
            if stage.name == name:
                return stage
        raise KeyError(name)

    def inbound(self, name: str) -> List[Edge]:
        """Return the edges pointing at ``name`` in declaration order."""
        return [edge for edge in self.edges if edge.target == name]

    def validate(self) -> None:
        """Raise on unknown references, forward cycles, or dead stages."""
        assert_references(self.stages, self.edges)
        assert_stage_exists(self.input, self.stages)
        assert_stage_exists(self.output, self.stages)
        assert_acyclic(self.stages, self.edges)
        assert_reachable(self.input, self.stages, self.edges)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-able dict for checkpoint serialization."""
        return {
            "stages": [stage.to_dict() for stage in self.stages],
            "edges": [edge.to_dict() for edge in self.edges],
            "input": self.input,
            "output": self.output,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TopologySpec":
        """Rebuild a spec from :meth:`to_dict` output."""
        return cls(
            stages=[Stage.from_dict(item) for item in data["stages"]],
            edges=[Edge.from_dict(item) for item in data["edges"]],
            input=str(data["input"]),
            output=str(data["output"]),
        )


def _branch_edges(
    items: Sequence[Stage], source: str, sink: str
) -> List[Edge]:
    """Return the edges linking ``source`` through ``items`` to ``sink``."""
    nodes = [source, *(stage.name for stage in items), sink]
    return [
        Edge(nodes[index], nodes[index + 1])
        for index in range(len(nodes) - 1)
    ]


def chain(
    stages: Sequence[Stage],
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> TopologySpec:
    """Link ``stages`` head-to-tail into a linear topology."""
    items = list(stages)
    if not items:
        raise ValueError("chain requires at least one stage")
    edges = [
        Edge(items[index].name, items[index + 1].name)
        for index in range(len(items) - 1)
    ]
    return TopologySpec(
        stages=items,
        edges=edges,
        input=input_name or items[0].name,
        output=output_name or items[-1].name,
    )


def residual(
    stages: Sequence[Stage],
    skip_from: str,
    skip_to: str,
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> TopologySpec:
    """Chain ``stages`` and add a weight-1 skip edge to a merge stage."""
    spec = chain(stages, input_name=input_name, output_name=output_name)
    return TopologySpec(
        stages=spec.stages,
        edges=[*spec.edges, Edge(skip_from, skip_to)],
        input=spec.input,
        output=spec.output,
    )


def multi_branch(
    input_stage: Stage,
    branches: Sequence[Sequence[Stage]],
    merge: Stage,
    output_name: Optional[str] = None,
) -> TopologySpec:
    """Fan ``input_stage`` into each branch and merge them at ``merge``."""
    groups = [list(branch) for branch in branches]
    stages: List[Stage] = [input_stage]
    edges: List[Edge] = []
    for items in groups:
        stages.extend(items)
        edges.extend(_branch_edges(items, input_stage.name, merge.name))
    stages.append(merge)
    return TopologySpec(
        stages=stages,
        edges=edges,
        input=input_stage.name,
        output=output_name or merge.name,
    )


def recurrent(
    stages: Sequence[Stage],
    feedback_from: str,
    feedback_to: str,
    input_name: Optional[str] = None,
    output_name: Optional[str] = None,
) -> TopologySpec:
    """Chain ``stages`` and add a one-step delayed feedback edge."""
    spec = chain(stages, input_name=input_name, output_name=output_name)
    return TopologySpec(
        stages=spec.stages,
        edges=[
            *spec.edges,
            Edge(feedback_from, feedback_to, delayed=True),
        ],
        input=spec.input,
        output=spec.output,
    )
