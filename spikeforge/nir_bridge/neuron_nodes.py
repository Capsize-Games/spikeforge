"""Render neuron stages into NIR nodes.

``leaky``/``lapicque`` are first-order integrate-and-fire neurons. The
installed ``nir.LIF`` applies a hard reset to ``v_reset`` and therefore
cannot keep the residual that snnTorch's ``subtract`` reset subtracts from
the membrane. To stay faithful rather than drift, ``subtract`` is rendered
as an ``nir.LI`` integrator plus a ``nir.Threshold`` spike node and a
one-step ``nir.Delay`` + ``nir.Scale`` feedback that injects ``-threshold``
into the integrator on the step after a spike. That reproduces
``mem[t] = beta * mem[t-1] + I[t] - threshold * spk[t-1]`` exactly.

A ``zero`` reset *is* exactly a hard reset to zero, so it uses ``nir.LIF``
directly; ``none`` is a plain ``nir.LI`` plus ``nir.Threshold``.

``synaptic`` maps to ``nir.CubaLIF`` because that node natively carries both
the synaptic and membrane time constants. ``recurrent`` maps to ``nir.LIF``
plus a ``nir.Delay`` node that feeds the previous output back into the
membrane; the delay breaks the cycle so the graph stays valid.

``alpha`` has no faithful rendering in the installed ``nir`` (it needs three
states and an alpha-function membrane, while ``nir.CubaLIF`` holds a single
synaptic current), so it is listed in :data:`UNMAPPABLE_KINDS` and export
raises the typed :class:`UnsupportedStageError` naming the kind instead of
inventing a lossy mapping. ``alpha`` stays available for simulation and
introspection.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np

from spikeforge.neurons import contract
from spikeforge.neurons.registry import NEURONS, handler
from spikeforge.nir_bridge import node_names
from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.mapped_node import MappedNode
from spikeforge.nir_bridge.neuron_kwargs import (
    cuba_kwargs,
    li_kwargs,
    lif_kwargs,
)
from spikeforge.nir_bridge.require import require_node
from spikeforge.nir_bridge.stage_mapping import StageMapping
from spikeforge.topology.stage import Stage

#: Suffix appended to a recurrent stage name for its feedback delay node.
FEEDBACK_SUFFIX = "__fb"
#: One timestep of feedback delay.
DELAY_STEPS = 1.0
#: Neuron kinds with no faithful NIR rendering in the installed ``nir``.
UNMAPPABLE_KINDS: Dict[str, str] = {
    "alpha": (
        "snn.Alpha keeps three states (syn_exc, syn_inh, mem) and an "
        "alpha-function membrane; the installed nir has no Alpha primitive "
        "and nir.CubaLIF carries only one synaptic current"
    ),
}

NeuronBuilder = Callable[[Stage], StageMapping]


def is_neuron_kind(kind: str) -> bool:
    """Return True when ``kind`` names a registered neuron."""
    return kind in NEURONS


def _scalar(value: Any) -> np.ndarray:
    """Return ``value`` as a scalar float32 array."""
    return np.array(float(value), dtype=np.float32)


def _single(name: str, node: Any) -> StageMapping:
    """Wrap a single node into a self-consuming stage mapping."""
    return StageMapping((MappedNode(name, node),), name, name, ())


def _integrator(name: str, kind: str, params: Dict[str, Any]) -> MappedNode:
    """Return the ``nir.LI`` integrator node for a first-order neuron."""
    return MappedNode(name, require_node("LI", kind)(**li_kwargs(params)))


def _threshold(name: str, kind: str, params: Dict[str, Any]) -> MappedNode:
    """Return the ``nir.Threshold`` spike node for a first-order neuron."""
    cls = require_node("Threshold", kind)
    return MappedNode(name, cls(_scalar(params["v_threshold"])))


def _reset_chain(name: str, kind: str, params: Dict[str, Any]) -> StageMapping:
    """Render a subtract reset as a delayed negative-threshold feedback."""
    mem = node_names.membrane_node(name, kind)
    delay = node_names.reset_delay_node(name)
    scale = node_names.reset_scale_node(name)
    delay_cls = require_node("Delay", kind)
    scale_cls = require_node("Scale", kind)
    nodes = (
        _integrator(mem, kind, params),
        _threshold(name, kind, params),
        MappedNode(delay, delay_cls(_scalar(DELAY_STEPS))),
        MappedNode(scale, scale_cls(_scalar(-float(params["v_threshold"])))),
    )
    edges: Tuple[Tuple[str, str], ...] = (
        (mem, name),
        (name, delay),
        (delay, scale),
        (scale, mem),
    )
    return StageMapping(nodes, mem, name, edges)


def _no_reset(name: str, kind: str, params: Dict[str, Any]) -> StageMapping:
    """Render a first-order neuron with no reset as LI + Threshold."""
    mem = node_names.membrane_node(name, kind)
    nodes = (_integrator(mem, kind, params), _threshold(name, kind, params))
    return StageMapping(nodes, mem, name, ((mem, name),))


def _zero_reset(name: str, kind: str, params: Dict[str, Any]) -> StageMapping:
    """Render a zero reset as a hard-reset ``nir.LIF`` node."""
    node = require_node("LIF", kind)(**lif_kwargs(params))
    return _single(name, node)


def _integrate_and_fire(stage: Stage) -> StageMapping:
    """Return the NIR rendering of a leaky/Lapicque stage."""
    params = handler(stage.kind).nir_params(stage.params)
    reset = contract.get_reset(stage.params)
    if reset == contract.RESET_ZERO:
        return _zero_reset(stage.name, stage.kind, params)
    if reset == contract.RESET_NONE:
        return _no_reset(stage.name, stage.kind, params)
    return _reset_chain(stage.name, stage.kind, params)


def _cuba(stage: Stage) -> StageMapping:
    """Return the ``nir.CubaLIF`` rendering of a synaptic stage."""
    cls = require_node("CubaLIF", stage.kind)
    params = handler(stage.kind).nir_params(stage.params)
    return _single(stage.name, cls(**cuba_kwargs(params)))


def _recurrent(stage: Stage) -> StageMapping:
    """Return a ``nir.LIF`` plus a ``nir.Delay`` feedback loop."""
    params = handler(stage.kind).nir_params(stage.params)
    lif = require_node("LIF", stage.kind)(**lif_kwargs(params))
    delay = require_node("Delay", stage.kind)(
        np.array(DELAY_STEPS, np.float32)
    )
    feedback = f"{stage.name}{FEEDBACK_SUFFIX}"
    nodes = (MappedNode(stage.name, lif), MappedNode(feedback, delay))
    edges = ((stage.name, feedback), (feedback, stage.name))
    return StageMapping(nodes, stage.name, stage.name, edges)


_BUILDERS: Dict[str, NeuronBuilder] = {
    "leaky": _integrate_and_fire,
    "lapicque": _integrate_and_fire,
    "synaptic": _cuba,
    "recurrent": _recurrent,
}


def neuron_mapping(stage: Stage) -> StageMapping:
    """Return the NIR rendering for a neuron ``stage``.

    A kind the installed ``nir`` cannot represent faithfully raises
    :class:`UnsupportedStageError` naming the kind, so export fails loudly
    rather than silently dropping or degrading the stage.
    """
    builder = _BUILDERS.get(stage.kind)
    if builder is not None:
        return builder(stage)
    if stage.kind in UNMAPPABLE_KINDS:
        raise UnsupportedStageError(stage.kind, UNMAPPABLE_KINDS[stage.kind])
    raise UnsupportedStageError(stage.kind)
