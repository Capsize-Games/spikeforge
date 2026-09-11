"""Inference on the displayed sample: prediction plus layer activity."""

from typing import Any, Dict, List, Mapping, Optional

import torch

from spikeforge.neurons.registry import NEURONS
from spikeforge.runtime.execution_mode import ExecutionMode
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.runner import run
from spikeforge.simulator.trajectory import Trajectory
from spikeforge.topology.spec import TopologySpec


def infer_spikes(
    net: Any,
    spikes: torch.Tensor,
    num_classes: int,
    true_label: Optional[int] = None,
    coding: str = "rate",
    hidden_cap: int = 256,
    membrane: bool = False,
    mode: ExecutionMode = ExecutionMode.PRODUCTION,
) -> Dict[str, Any]:
    """Score one sample's encoded spikes and summarise its activity."""
    trajectory = run(
        net, spikes, track=True, membrane=membrane, mode=mode
    )
    spec = spec_of(net)
    output = _frames(trajectory.spikes[spec.output])
    hidden = _frames(trajectory.spikes[hidden_stage(spec)])
    payload = _scalars(trajectory, output, true_label, coding)
    payload = _add_membrane(payload, trajectory, membrane)
    return _with_rasters(payload, hidden, output, num_classes, hidden_cap)


def _add_membrane(
    payload: Dict[str, Any], trajectory: Trajectory, membrane: bool
) -> Dict[str, Any]:
    """Attach membrane traces when requested, else return ``payload``.

    ``membrane=True`` adds a ``membrane`` key holding per-step traces per
    neuron stage; every existing payload key is left exactly as it was.
    """
    if membrane:
        payload["membrane"] = _membrane_traces(trajectory.membranes)
    return payload


def hidden_stage(spec: TopologySpec) -> str:
    """Return the neuron stage whose spikes precede ``spec.output``.

    Locating it from the spec (rather than hard-coding ``_lif1``) is what
    lets every topology report a meaningful hidden raster.
    """
    previous: Optional[str] = None
    for stage in spec.stages:
        if stage.name == spec.output:
            break
        if stage.kind in NEURONS:
            previous = stage.name
    if previous is None:
        raise ValueError("topology has no neuron stage before its output")
    return previous


def _scalars(
    trajectory: Trajectory,
    output: List[torch.Tensor],
    true_label: Optional[int],
    coding: str,
) -> Dict[str, Any]:
    """Assemble the prediction scalars and per-class readout."""
    logits = trajectory.logits[0].detach()
    probs = torch.softmax(logits, dim=0)
    return {
        "predicted": int(logits.argmax()),
        "confidence": float(probs.max()),
        "true_label": None if true_label is None else int(true_label),
        "class_spikes": [float(x) for x in _class_totals(output)],
        "output_over_time": _over_time(output),
        "coding": coding,
        "input_mode": coding,
        "num_steps": int(trajectory.steps),
    }


def _frames(trace: torch.Tensor) -> List[torch.Tensor]:
    """Split a ``[T, ...]`` trace tensor into a list of per-step frames."""
    return [trace[index] for index in range(trace.size(0))]


def _with_rasters(
    payload: Dict[str, Any],
    hidden: List[torch.Tensor],
    output: List[torch.Tensor],
    num_classes: int,
    hidden_cap: int,
) -> Dict[str, Any]:
    """Attach bounded hidden/output rasters under a private payload key."""
    payload["_rasters"] = {
        "hidden": layer_raster(hidden, hidden_cap),
        "output": layer_raster(output, num_classes),
    }
    return payload


def layer_raster(
    frames: List[torch.Tensor], max_neurons: int
) -> Dict[str, Any]:
    """Collapse tracked per-step frames into a bounded ``[T, N]`` raster.

    Each frame's first sample is flattened, so feature and spatial neuron
    stages share one coordinate convention.
    """
    stacked = [frame[0].reshape(-1) for frame in frames]
    matrix = torch.stack(stacked).detach().cpu()
    matrix = matrix[:, : max(1, int(max_neurons))]
    time_idx, neuron_idx = torch.where(matrix > 0)
    return {
        "time": time_idx.tolist(),
        "neurons": neuron_idx.tolist(),
        "num_steps": int(matrix.size(0)),
        "num_neurons": int(matrix.size(1)),
    }


def _class_totals(output: List[torch.Tensor]) -> torch.Tensor:
    """Sum output spikes over time into a per-class total."""
    stacked = torch.stack([frame[0] for frame in output])
    return stacked.sum(dim=0).detach()


def _over_time(output: List[torch.Tensor]) -> List[List[float]]:
    """Return the ``[T][num_classes]`` output spike matrix for sample 0."""
    return [[float(x) for x in frame[0].detach()] for frame in output]


def _membrane_traces(
    traces: Mapping[str, torch.Tensor],
) -> Dict[str, List[List[float]]]:
    """Return per-stage ``[T][N]`` membrane voltages for sample 0."""
    return {
        name: _rows(trace) for name, trace in traces.items()
    }


def _rows(trace: torch.Tensor) -> List[List[float]]:
    """Return per-step flattened sample-0 frames as plain floats."""
    return [
        [float(x) for x in frame[0].detach().reshape(-1)]
        for frame in _frames(trace)
    ]
