"""Registry of selectable topologies and their default parameters.

This is the single lookup shared by the training engine, the verify CLI, and
later phases: it maps a topology name to its spec builder plus the parameter
names that builder accepts. :func:`build_topology` returns the rendered
``(spec, module)`` pair, and :func:`resolved_params` returns the effective
mapping after defaults are merged with caller overrides.

The ``neurons`` (per-stage kind) and ``stage_params`` (per-stage params)
overrides are part of every preset's default mapping, so callers can forward
them like any other override without special-casing a topology.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from spikeforge.topology import presets, sequence_presets
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

SpecBuilder = Callable[..., TopologySpec]
ModuleBuilder = Callable[[Mapping[str, Any]], StageModule]

#: Neuron kind selected by every preset when ``params["neuron"]`` is absent.
_DEFAULT_NEURON = presets.DEFAULT_NEURON
#: Default surrogate selection (``None`` keeps snnTorch's own default).
_DEFAULT_SURROGATE: Optional[str] = None
#: The per-stage override keys every neuron preset accepts.
_OVERRIDE_KEYS: Dict[str, Any] = {"neurons": None, "stage_params": None}

_LEGACY_DEFAULTS: Dict[str, Any] = {
    "hidden": 128,
    "beta": 0.5,
    "num_classes": 10,
    "input_size": 28 * 28,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    **_OVERRIDE_KEYS,
}
_SMALL_DEFAULTS: Dict[str, Any] = {
    "hidden": 32,
    "beta": 0.9,
    "num_classes": 10,
    "input_size": 28 * 28,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    **_OVERRIDE_KEYS,
}
_CONV_DEFAULTS: Dict[str, Any] = {
    "in_channels": 1,
    "channels": 8,
    "num_classes": 10,
    "input_size": 28,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    **_OVERRIDE_KEYS,
}
_RECURRENT_DEFAULTS: Dict[str, Any] = {
    "hidden": 64,
    "beta": 0.9,
    "num_classes": 10,
    "input_size": 28 * 28,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    **_OVERRIDE_KEYS,
}
_SEQUENCE_MLP_DEFAULTS: Dict[str, Any] = {
    "seq_length": 8,
    "features": 8,
    "hidden": 16,
    "beta": 0.9,
    "num_classes": 4,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    "threshold": None,
    "reset": "zero",
    **_OVERRIDE_KEYS,
}
_SEQUENCE_ATTN_DEFAULTS: Dict[str, Any] = {
    "seq_length": 8,
    "vocab": 32,
    "embed_dim": 16,
    "num_heads": 2,
    "beta": 0.9,
    "num_classes": 4,
    "neuron": _DEFAULT_NEURON,
    "surrogate": _DEFAULT_SURROGATE,
    "threshold": None,
    "reset": "subtract",
    **_OVERRIDE_KEYS,
}

_Entry = Tuple[SpecBuilder, Mapping[str, Any], Optional[ModuleBuilder]]


def _legacy_module(params: Mapping[str, Any]) -> StageModule:
    """Render ``fc_legacy`` params as the legacy ``SpikingNet`` wrapper.

    Imported lazily because ``SpikingNet`` itself depends on the presets, so
    a module-level import here would be circular. Keeping the legacy wrapper
    is what preserves ``forward_spikes`` and the original state-dict keys.
    """
    from spikeforge.network.spiking_net import SpikingNet

    return SpikingNet(
        hidden=int(params["hidden"]),
        beta=float(params["beta"]),
        num_classes=int(params["num_classes"]),
        input_size=int(params["input_size"]),
    )


_REGISTRY: Dict[str, _Entry] = {
    "fc_legacy": (presets.fc_legacy, _LEGACY_DEFAULTS, _legacy_module),
    "fc_small": (presets.fc_small, _SMALL_DEFAULTS, None),
    "conv_net": (presets.conv_net, _CONV_DEFAULTS, None),
    "recurrent_net": (presets.recurrent_net, _RECURRENT_DEFAULTS, None),
    "sequence_mlp": (
        sequence_presets.sequence_mlp, _SEQUENCE_MLP_DEFAULTS, None,
    ),
    "sequence_attn": (
        sequence_presets.sequence_attn, _SEQUENCE_ATTN_DEFAULTS, None,
    ),
}

#: Topologies whose input frames carry a ``[B, L, D]`` sequence layout.
_SEQUENCE_NAMES = frozenset({"sequence_mlp", "sequence_attn"})


def _entry(name: str) -> _Entry:
    """Return the registry entry for ``name`` or raise ``ValueError``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown topology: {name!r}") from None


def topology_names() -> List[str]:
    """Return the selectable topology names in registry order."""
    return list(_REGISTRY)


def is_sequence_topology(name: str) -> bool:
    """Return True when ``name`` consumes a sequence-layout input frame."""
    return name in _SEQUENCE_NAMES


def resolved_params(
    name: str, params: Optional[Mapping[str, Any]] = None
) -> Dict[str, Any]:
    """Return ``name``'s defaults merged with the accepted ``params``.

    Keys the topology's builder does not understand are ignored, so callers
    can pass engine-wide arguments (``hidden``, ``num_classes``, ...) to every
    topology without special-casing the ones that do not use them.
    """
    defaults = dict(_entry(name)[1])
    for key, value in (params or {}).items():
        if key in defaults:
            defaults[key] = value
    return defaults


def _uses_legacy_wrapper(resolved: Mapping[str, Any]) -> bool:
    """Return True when ``fc_legacy`` can still render its legacy wrapper.

    The ``SpikingNet`` wrapper implements only ``snn.Leaky`` and takes no
    surrogate, so the generic module is used whenever the neuron, surrogate,
    or any per-stage override departs from the historical default.
    """
    default_neuron = resolved.get("neuron", _DEFAULT_NEURON)
    overrides = resolved.get("neurons") or {}
    stage_params = resolved.get("stage_params") or {}
    return (
        default_neuron == _DEFAULT_NEURON
        and resolved.get("surrogate") is None
        and not overrides
        and not stage_params
    )


def build_topology(
    name: str, params: Optional[Mapping[str, Any]] = None
) -> Tuple[TopologySpec, StageModule]:
    """Build ``name`` and return its ``(spec, module)`` pair."""
    builder, _, module_builder = _entry(name)
    resolved = resolved_params(name, params)
    spec = builder(**resolved)
    if module_builder is not None and _uses_legacy_wrapper(resolved):
        return spec, module_builder(resolved)
    return spec, build_module(spec)
