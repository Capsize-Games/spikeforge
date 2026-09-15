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


def expected_state(
    name: str, num_classes: Optional[int] = None
) -> Optional[Dict[str, Tuple[int, ...]]]:
    """Return a preset module's ``key -> shape`` mapping, if it can build.

    ``num_classes`` builds the comparison preset with the class count the
    artifact declares instead of the preset's default. Without it, every
    checkpoint trained on a dataset that is not 10-class reports a readout
    shape mismatch that says nothing about compatibility -- only that the
    comparison was built against the wrong output size.
    """
    from spikeforge.topology.registry import build_topology

    params = {} if num_classes is None else {"num_classes": int(num_classes)}
    try:
        _spec, module = build_topology(name, params)
        state = module.state_dict()
    except _ERRORS:
        return None
    return {key: tuple(value.shape) for key, value in state.items()}
