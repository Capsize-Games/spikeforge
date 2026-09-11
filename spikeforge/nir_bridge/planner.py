"""Plan the ordered NIR nodes and edges that render a topology spec.

The planner is where the graph structure is decided. Parameterless merge
stages (``add``) emit no node: their predecessors are wired straight to
their successors, so a fan-in is expressed as several edges converging on
one node. Delayed spec edges gain a synthetic ``nir.Delay`` node so the
exported graph encodes recurrence the NIR way. A stage that expands into
several nodes contributes its own internal edges, which is how a neuron
feeds its reset path back into its own integrator.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from spikeforge.nir_bridge.mapped_node import MappedNode
from spikeforge.nir_bridge.mapper import map_stage
from spikeforge.nir_bridge.require import require_node
from spikeforge.nir_bridge.stage_mapping import StageMapping
from spikeforge.topology.spec import TopologySpec

#: Reserved node names for the graph's external entry and exit.
INPUT_NODE = "input"
OUTPUT_NODE = "output"
#: Tag inserted into a synthetic delay node's name.
DELAY_MARK = "__delay__"
#: One timestep of delay carried by a feedback edge.
DELAY_STEPS = 1.0

#: ``(node, delayed)`` pairs carrying a stage's output.
Outputs = Tuple[Tuple[str, bool], ...]
#: The ordered nodes and the resolved edges of an export.
Plan = Tuple[Tuple[MappedNode, ...], Tuple[Tuple[str, str], ...]]


def _map_stages(
    spec: TopologySpec, module: Optional[Any]
) -> Dict[str, StageMapping]:
    """Map every stage into its NIR rendering, in declaration order."""
    return {stage.name: map_stage(stage, module) for stage in spec.stages}


def _merge_outputs(
    name: str,
    spec: TopologySpec,
    mappings: Mapping[str, StageMapping],
    memo: Dict[str, Outputs],
    active: Set[str],
) -> Outputs:
    """Resolve a merge stage's output to its inbound nodes."""
    if name in active:
        raise ValueError(f"delayed cycle passes through merge stage {name!r}")
    active.add(name)
    items = _merge_items(name, spec, mappings, memo, active)
    active.discard(name)
    result: Outputs = tuple(dict.fromkeys(items))
    memo[name] = result
    return result


def _merge_items(
    name: str,
    spec: TopologySpec,
    mappings: Mapping[str, StageMapping],
    memo: Dict[str, Outputs],
    active: Set[str],
) -> List[Tuple[str, bool]]:
    """Collect the inbound ``(node, delayed)`` pairs for a merge stage."""
    items: List[Tuple[str, bool]] = []
    if name == spec.input:
        items.append((INPUT_NODE, False))
    for edge in spec.inbound(name):
        for node, delayed in _outputs(
            edge.source, spec, mappings, memo, active
        ):
            items.append((node, delayed or edge.delayed))
    return items


def _outputs(
    name: str,
    spec: TopologySpec,
    mappings: Mapping[str, StageMapping],
    memo: Dict[str, Outputs],
    active: Set[str],
) -> Outputs:
    """Return the ``(node, delayed)`` pairs carrying ``name``'s output."""
    cached = memo.get(name)
    if cached is not None:
        return cached
    mapping = mappings[name]
    if mapping.output is not None:
        result: Outputs = ((mapping.output, False),)
        memo[name] = result
        return result
    return _merge_outputs(name, spec, mappings, memo, active)


def _internal_edges(
    mappings: Mapping[str, StageMapping],
) -> List[Tuple[str, str, bool]]:
    """Return the edges internal to multi-node stage renderings."""
    return [
        (source, target, False)
        for mapping in mappings.values()
        for source, target in mapping.edges
    ]


def _collect_edges(
    spec: TopologySpec, mappings: Mapping[str, StageMapping]
) -> List[Tuple[str, str, bool]]:
    """Return every logical ``(source, target, delayed)`` edge."""
    memo: Dict[str, Outputs] = {}
    edges: List[Tuple[str, str, bool]] = []
    for stage in spec.stages:
        mapping = mappings[stage.name]
        if mapping.entry is None:
            continue
        if stage.name == spec.input:
            edges.append((INPUT_NODE, mapping.entry, False))
        for edge in spec.inbound(stage.name):
            for node, delayed in _outputs(
                edge.source, spec, mappings, memo, set()
            ):
                edges.append((node, mapping.entry, delayed or edge.delayed))
    for node, delayed in _outputs(spec.output, spec, mappings, memo, set()):
        edges.append((node, OUTPUT_NODE, delayed))
    return list(dict.fromkeys([*edges, *_internal_edges(mappings)]))


def _delay_name(source: str, target: str, counts: Dict[str, int]) -> str:
    """Return a unique name for the delay node on ``source -> target``."""
    base = f"{source}{DELAY_MARK}{target}"
    index = counts.get(base, 0)
    counts[base] = index + 1
    return base if index == 0 else f"{base}__{index}"


def _insert_delays(
    edges: Sequence[Tuple[str, str, bool]],
) -> Tuple[List[MappedNode], List[Tuple[str, str]]]:
    """Replace each delayed edge with an intermediate ``Delay`` node."""
    delay_cls = require_node("Delay", "delay")
    counts: Dict[str, int] = {}
    nodes: List[MappedNode] = []
    resolved: List[Tuple[str, str]] = []
    for source, target, delayed in edges:
        if not delayed:
            resolved.append((source, target))
            continue
        name = _delay_name(source, target, counts)
        node = delay_cls(np.array(DELAY_STEPS, np.float32))
        nodes.append(MappedNode(name, node))
        resolved.append((source, name))
        resolved.append((name, target))
    return nodes, resolved


def _input_node() -> MappedNode:
    """Return the graph's ``Input`` node."""
    cls = require_node("Input", "input")
    return MappedNode(INPUT_NODE, cls({"input": None}))


def _output_node() -> MappedNode:
    """Return the graph's ``Output`` node."""
    cls = require_node("Output", "output")
    return MappedNode(OUTPUT_NODE, cls({"output": None}))


def build_plan(
    spec: TopologySpec, module: Optional[Any] = None
) -> Plan:
    """Return the ordered nodes and resolved edges rendering ``spec``."""
    spec.validate()
    mappings = _map_stages(spec, module)
    edges = _collect_edges(spec, mappings)
    delay_nodes, resolved = _insert_delays(edges)
    stage_nodes = [
        node for mapping in mappings.values() for node in mapping.nodes
    ]
    nodes = (_input_node(), *stage_nodes, *delay_nodes, _output_node())
    return nodes, tuple(resolved)
