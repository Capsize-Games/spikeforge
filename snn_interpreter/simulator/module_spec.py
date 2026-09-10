"""Access to the topology spec a runnable module was rendered from."""

from snn_interpreter.topology.spec import TopologySpec


def spec_of(module: object) -> TopologySpec:
    """Return the :class:`TopologySpec` driving ``module``.

    A ``StageModule`` keeps its spec as the private ``_spec`` attribute and
    exposes no other handle on the graph, so the simulator reads it here.
    This keeps the public ``run`` signature module-only.
    """
    spec = module._spec
    if not isinstance(spec, TopologySpec):
        raise TypeError("module was not built from a TopologySpec")
    return spec
