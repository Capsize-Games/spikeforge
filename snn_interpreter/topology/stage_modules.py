"""Factories rendering parameterised stage kinds into torch modules."""

from typing import Any, Callable, Dict, Mapping, Optional

import torch.nn as nn

from snn_interpreter.neurons.registry import NEURONS, build_neuron
from snn_interpreter.topology.kinds import PARAMETERLESS_KINDS
from snn_interpreter.topology.stage import Stage
from snn_interpreter.topology.sum_pool import SumPool2d

StageBuilder = Callable[[Mapping[str, Any]], nn.Module]


def _linear(params: Mapping[str, Any]) -> nn.Module:
    return nn.Linear(
        int(params["in_features"]),
        int(params["out_features"]),
        bias=bool(params.get("bias", True)),
    )


def _flatten(params: Mapping[str, Any]) -> nn.Module:
    return nn.Flatten(
        start_dim=int(params.get("start_dim", 1)),
        end_dim=int(params.get("end_dim", -1)),
    )


def _conv2d(params: Mapping[str, Any]) -> nn.Module:
    return nn.Conv2d(
        int(params["in_channels"]),
        int(params["out_channels"]),
        kernel_size=params["kernel_size"],
        stride=params.get("stride", 1),
        padding=params.get("padding", 0),
        dilation=params.get("dilation", 1),
        groups=int(params.get("groups", 1)),
        bias=bool(params.get("bias", True)),
    )


def _pool_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "kernel_size": params["kernel_size"],
        "stride": params.get("stride"),
        "padding": params.get("padding", 0),
    }


def _avgpool2d(params: Mapping[str, Any]) -> nn.Module:
    return nn.AvgPool2d(**_pool_kwargs(params))


def _sumpool2d(params: Mapping[str, Any]) -> nn.Module:
    return SumPool2d(**_pool_kwargs(params))


_BUILDERS: Dict[str, StageBuilder] = {
    "linear": _linear,
    "flatten": _flatten,
    "conv2d": _conv2d,
    "avgpool2d": _avgpool2d,
    "sumpool2d": _sumpool2d,
}


def is_neuron_kind(kind: str) -> bool:
    """Return True when ``kind`` is a registered neuron kind."""
    return kind in NEURONS


def module_for_stage(stage: Stage) -> Optional[nn.Module]:
    """Return the module rendering ``stage``, or ``None`` if parameter-free."""
    if is_neuron_kind(stage.kind):
        return build_neuron(stage.kind, stage.params)
    if stage.kind in PARAMETERLESS_KINDS:
        return None
    builder = _BUILDERS.get(stage.kind)
    if builder is None:
        raise ValueError(f"unknown stage kind: {stage.kind!r}")
    return builder(stage.params)
