"""The node(s) and wiring a single substitution rule produces.

One rewrite rule has a uniform result shape so a one-for-one replacement
(``IF`` -> ``LIF``) and a multi-node expansion (``AvgPool2d`` -> ``SumPool2d``
plus ``Scale``) share the same interface. ``entry`` is the node that now
consumes the replaced node's inputs and ``output`` the node that now carries
its result, which is how the executor rewires the surrounding edges.
"""

from typing import Any, NamedTuple, Tuple

#: ``(node name, nir node)`` pairs emitted by a rule.
RewriteNodes = Tuple[Tuple[str, Any], ...]
#: Internal ``(source, target)`` edges of a multi-node expansion.
RewriteEdges = Tuple[Tuple[str, str], ...]


class RewriteOutcome(NamedTuple):
    """The nodes, internal edges, and wiring handles of one rewrite."""

    nodes: RewriteNodes
    edges: RewriteEdges
    entry: str
    output: str
    detail: str
