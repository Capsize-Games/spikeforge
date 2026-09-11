"""One NIR node produced by mapping a single topology stage."""

from typing import Any, NamedTuple


class MappedNode(NamedTuple):
    """A named NIR node, kept alongside the stage it renders.

    ``node`` is deliberately typed as ``Any``: the concrete class comes from
    the installed ``nir`` package and is resolved through
    :mod:`spikeforge.nir_bridge.api`.
    """

    name: str
    node: Any
