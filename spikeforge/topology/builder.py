"""Build a runnable :class:`StageModule` from a :class:`TopologySpec`."""

from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule


def build_module(spec: TopologySpec) -> StageModule:
    """Return a runnable module rendering of ``spec``."""
    return StageModule(spec)
