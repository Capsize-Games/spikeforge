"""The NIR nodes a single stage expands into, plus its wiring handles."""

from typing import NamedTuple, Optional, Tuple

from spikeforge.nir_bridge.mapped_node import MappedNode


class StageMapping(NamedTuple):
    """Describe how one stage renders into NIR.

    ``nodes`` are the emitted nodes. ``entry`` is the node that consumes the
    stage's incoming signal and ``output`` the node that carries the stage's
    result; both are ``None`` for a parameterless merge stage, which is
    resolved by connecting its predecessors straight to its successors.
    ``edges`` are the internal edges of a multi-node expansion.
    """

    nodes: Tuple[MappedNode, ...]
    entry: Optional[str]
    output: Optional[str]
    edges: Tuple[Tuple[str, str], ...]
