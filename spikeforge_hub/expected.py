"""The preset structure an inspected artifact is compared against.

Isolating preset rendering keeps :mod:`spikeforge_hub.compat` free of the
NIR and torch build details it only needs indirectly. Every expectation is
best-effort: a preset that cannot be rendered returns ``None`` so the caller
skips it rather than failing the whole classification.
"""

from typing import Any, Dict, List, Optional, Tuple

from spikeforge.nir_bridge.errors import UnsupportedStageError

#: Errors that mean a preset simply cannot be rendered for comparison.
_ERRORS = (ValueError, RuntimeError, OSError, UnsupportedStageError)


def candidates(topology: Optional[str]) -> List[str]:
    """Return the preset names to compare, or just the requested one."""
    if topology is not None:
        return [topology]
    from spikeforge.topology.registry import topology_names

    return topology_names()


def expected_graph(name: str) -> Optional[Dict[str, Any]]:
    """Return the structural NIR summary a preset should render, if it can."""
    from spikeforge.nir_bridge import graph_summary
    from spikeforge.topology.registry import build_topology

    try:
        spec, _module = build_topology(name)
        return graph_summary(spec)
    except _ERRORS:
        return None


def expected_state(name: str) -> Optional[Dict[str, Tuple[int, ...]]]:
    """Return a preset module's ``key -> shape`` mapping, if it can build."""
    from spikeforge.topology.registry import build_topology

    try:
        _spec, module = build_topology(name)
        state = module.state_dict()
    except _ERRORS:
        return None
    return {key: tuple(value.shape) for key, value in state.items()}
