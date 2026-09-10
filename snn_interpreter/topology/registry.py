"""Registry of selectable topologies and their default parameters.

This is the single lookup shared by the training engine, the verify CLI, and
later phases: it maps a topology name to its spec builder plus the parameter
names that builder accepts. :func:`build_topology` returns the rendered
``(spec, module)`` pair, and :func:`resolved_params` returns the effective
mapping after defaults are merged with caller overrides.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from snn_interpreter.topology import presets
from snn_interpreter.topology.builder import build_module
from snn_interpreter.topology.spec import TopologySpec
from snn_interpreter.topology.stage_module import StageModule

SpecBuilder = Callable[..., TopologySpec]
ModuleBuilder = Callable[[Mapping[str, Any]], StageModule]

_LEGACY_DEFAULTS: Dict[str, Any] = {
    "hidden": 128,
    "beta": 0.5,
    "num_classes": 10,
    "input_size": 28 * 28,
}
_SMALL_DEFAULTS: Dict[str, Any] = {
    "hidden": 32,
    "beta": 0.9,
    "num_classes": 10,
    "input_size": 28 * 28,
}
_CONV_DEFAULTS: Dict[str, Any] = {
    "in_channels": 1,
    "channels": 8,
    "num_classes": 10,
    "input_size": 28,
}
_RECURRENT_DEFAULTS: Dict[str, Any] = {
    "hidden": 64,
    "beta": 0.9,
    "num_classes": 10,
    "input_size": 28 * 28,
}

_Entry = Tuple[SpecBuilder, Mapping[str, Any], Optional[ModuleBuilder]]


def _legacy_module(params: Mapping[str, Any]) -> StageModule:
    """Render ``fc_legacy`` params as the legacy ``SpikingNet`` wrapper.

    Imported lazily because ``SpikingNet`` itself depends on the presets, so
    a module-level import here would be circular. Keeping the legacy wrapper
    is what preserves ``forward_spikes`` and the original state-dict keys.
    """
    from snn_interpreter.network.spiking_net import SpikingNet

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
}


def _entry(name: str) -> _Entry:
    """Return the registry entry for ``name`` or raise ``ValueError``."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown topology: {name!r}") from None


def topology_names() -> List[str]:
    """Return the selectable topology names in registry order."""
    return list(_REGISTRY)


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


def build_topology(
    name: str, params: Optional[Mapping[str, Any]] = None
) -> Tuple[TopologySpec, StageModule]:
    """Build ``name`` and return its ``(spec, module)`` pair."""
    builder, _, module_builder = _entry(name)
    resolved = resolved_params(name, params)
    spec = builder(**resolved)
    if module_builder is not None:
        return spec, module_builder(resolved)
    return spec, build_module(spec)
