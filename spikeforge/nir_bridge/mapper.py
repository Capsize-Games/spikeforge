"""Translate a single topology stage into its NIR node rendering."""

from typing import Any, Callable, Dict, Mapping, Optional

from spikeforge.nir_bridge import (
    node_builders,
    stage_builders,
    stages_unmappable,
)
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.mapped_node import MappedNode
from spikeforge.nir_bridge.neuron_nodes import (
    is_neuron_kind,
    neuron_mapping,
)
from spikeforge.nir_bridge.stage_mapping import StageMapping
from spikeforge.topology.stage import Stage

#: Parameterless merge kind; it resolves to its predecessors' nodes.
PASSTHROUGH_KIND = "add"
#: Kinds that are the identity at inference and emit no node.
PASSTHROUGH_KINDS = frozenset({"dropout"})

NodeBuilder = Callable[[str, Mapping[str, Any], Optional[Any]], MappedNode]

_BUILDERS: Dict[str, NodeBuilder] = {
    "linear": node_builders.linear_node,
    "conv2d": node_builders.conv_node,
    "flatten": node_builders.flatten_node,
    "avgpool2d": node_builders.avgpool_node,
    "sumpool2d": node_builders.sumpool_node,
    **stage_builders.STAGE_BUILDERS,
}

_PASSTHROUGH = StageMapping((), None, None, ())


def _submodule(module: Optional[Any], name: str) -> Optional[Any]:
    """Return the submodule named ``name`` from ``module`` if present."""
    if module is None:
        return None
    try:
        return module.get_submodule(name)
    except AttributeError:
        return None


def map_stage(stage: Stage, module: Optional[Any] = None) -> StageMapping:
    """Return the NIR rendering of ``stage``.

    ``module`` is the optional built ``StageModule``; the submodule named
    after the stage supplies learned weights. Without it, weighted stages
    render with zero-initialised placeholder weights. A kind that cannot be
    mapped raises :class:`UnsupportedStageError` naming the kind and the
    reason the installed ``nir`` cannot represent it.
    """
    builder = _BUILDERS.get(stage.kind)
    if builder is not None:
        submodule = _submodule(module, stage.name)
        node = builder(stage.name, stage.params, submodule)
        return StageMapping((node,), stage.name, stage.name, ())
    if stage.kind == PASSTHROUGH_KIND or stage.kind in PASSTHROUGH_KINDS:
        return _PASSTHROUGH
    if is_neuron_kind(stage.kind):
        return neuron_mapping(stage)
    reason = stages_unmappable.UNMAPPABLE_STAGES.get(stage.kind, "")
    raise UnsupportedStageError(stage.kind, reason)
