"""NIR builder table and live contract probe for the extended stage kinds.

Only ``conv1d`` has a faithful NIR primitive among the new kinds. Its contract
is probed against the installed ``nir`` at call time rather than hardcoded, so
a future ``nir`` that adds or drops ``Conv1d`` flips the reported outcome
honestly instead of silently drifting.
"""

from typing import Any, Callable, Dict, Mapping, Optional

from spikeforge.nir_bridge import api, node_builders
from spikeforge.nir_bridge.mapped_node import MappedNode

#: NIR contract outcomes a stage kind can declare.
MAPPED = "mapped"
PASSTHROUGH = "passthrough"
UNEXPORTABLE = "unexportable"

NodeBuilder = Callable[
    [str, Mapping[str, Any], Optional[Any]], MappedNode
]

#: Builders for the extended kinds that have a faithful NIR primitive.
STAGE_BUILDERS: Dict[str, NodeBuilder] = {
    "conv1d": node_builders.conv1d_node,
}


def conv1d_contract() -> str:
    """Return the conv1d contract: ``mapped`` or ``unexportable``."""
    if api.node_class("Conv1d") is None:
        return UNEXPORTABLE
    return MAPPED
