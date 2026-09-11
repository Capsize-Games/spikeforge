"""Run one spike train through every registered neuron kind side by side."""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge.neurons.registry import neuron_kinds
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec, chain
from spikeforge.topology.stage import Stage

#: Extra constructor defaults needed to build a kind without further input.
_KIND_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "alpha": {"alpha": 0.9, "beta": 0.8},
}


def _stage_params(
    kind: str, features: int, params: Mapping[str, Any]
) -> Dict[str, Any]:
    """Return the neuron stage params for ``kind``, filling its defaults."""
    merged: Dict[str, Any] = dict(_KIND_DEFAULTS.get(kind, {}))
    merged.update(params)
    if kind == "recurrent" and "linear_features" not in merged:
        merged["linear_features"] = features
    return merged


def _spec_for(
    kind: str, features: int, params: Mapping[str, Any]
) -> TopologySpec:
    """Return a linear-then-neuron chain driven by the shared input."""
    stages = [
        Stage(
            "fc",
            "linear",
            {"in_features": features, "out_features": features},
        ),
        Stage("neuron", kind, _stage_params(kind, features, params)),
    ]
    return chain(stages)


def _trajectory_for(
    kind: str,
    spikes: torch.Tensor,
    features: int,
    params: Mapping[str, Any],
    seed: int,
) -> Trajectory:
    """Build and run one neuron kind's chain on ``spikes``."""
    torch.manual_seed(seed)
    module = build_module(_spec_for(kind, features, params))
    return run(module, spikes, track=True, membrane=True, current=True)


def compare_neurons(
    spikes: torch.Tensor,
    kinds: Optional[Tuple[str, ...]] = None,
    params: Optional[Mapping[str, Mapping[str, Any]]] = None,
    seed: int = 0,
) -> Dict[str, Trajectory]:
    """Return ``{kind: Trajectory}`` for the same input spike train.

    ``params`` overrides constructor args per kind.
    """
    sample = spikes[0]
    features = int(sample.reshape(sample.size(0), -1).size(1))
    selected = kinds or neuron_kinds()
    overrides = dict(params or {})
    result: Dict[str, Trajectory] = {}
    for kind in selected:
        result[kind] = _trajectory_for(
            kind, spikes, features, overrides.get(kind, {}), seed
        )
    return result
