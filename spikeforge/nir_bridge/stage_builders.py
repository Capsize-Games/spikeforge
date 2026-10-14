"""NIR builder table and live contract probe for the extended stage kinds.

``conv1d`` maps to its own primitive, probed against the installed ``nir``
at call time rather than hardcoded, so a future ``nir`` that adds or drops
``Conv1d`` flips the reported outcome honestly instead of silently
drifting. ``embedding`` and ``batch_norm`` decompose into the
C-supported ``Affine`` primitive instead (see
:mod:`spikeforge.nir_bridge.affine_folds`).
"""

from typing import Any, Callable, Dict, Mapping, Optional

from spikeforge.nir_bridge import affine_folds, api, node_builders
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
    "embedding": affine_folds.embedding_node,
    "batch_norm": affine_folds.batch_norm_node,
}


def conv1d_contract() -> str:
    """Return the conv1d contract: ``mapped`` or ``unexportable``."""
    if api.node_class("Conv1d") is None:
        return UNEXPORTABLE
    return MAPPED
