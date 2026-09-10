"""Bounded JSON-able payloads for the Phase 2 introspection actions.

``trajectory`` capture is deliberately capped so a payload stays small even
for wide hidden layers or many stages: :data:`MAX_STAGES` bounds how many
neuron stages are reported, and :data:`MAX_NEURONS` bounds the feature width
of every ``U[t]``/``I[t]``/``S[t]`` row. This mirrors the raster helpers,
which bound neurons the same way. ``metrics`` aggregates the same
educational run, so its size is independent of those caps.
"""

from dataclasses import replace
from typing import Any, Dict, List

import torch

from server.schemas import TrainConfig
from snn_interpreter.benchmark import BenchmarkConfig, default_config
from snn_interpreter.benchmark.harness import run_benchmark
from snn_interpreter.introspection.metrics import trajectory_metrics
from snn_interpreter.runtime.execution_mode import ExecutionMode
from snn_interpreter.simulator import input_shape
from snn_interpreter.simulator.runner import run
from snn_interpreter.simulator.trajectory import Trajectory
from snn_interpreter.topology.spec import TopologySpec

#: At most this many neuron stages appear in a trajectory payload.
MAX_STAGES = 8
#: At most this many neurons per stage appear in a trajectory payload.
MAX_NEURONS = 64


def _width(trace: torch.Tensor) -> int:
    """Return the per-step feature count of a ``[T, B, ...]`` trace."""
    return int(trace[0][0].numel())


def _rows(trace: torch.Tensor, max_neurons: int) -> List[List[float]]:
    """Return per-step sample-0 rows capped to ``max_neurons`` features."""
    rows: List[List[float]] = []
    for index in range(int(trace.size(0))):
        flat = trace[index][0].detach().reshape(-1)[:max_neurons]
        rows.append([float(value) for value in flat])
    return rows


def _stage_names(trajectory: Trajectory) -> List[str]:
    """Return the recorded neuron stages, capped to the first MAX_STAGES."""
    return sorted(trajectory.spikes)[:MAX_STAGES]


def _neuron_counts(
    trajectory: Trajectory, names: List[str]
) -> Dict[str, int]:
    """Return the recorded feature width of each capped stage."""
    return {
        name: min(MAX_NEURONS, _width(trajectory.spikes[name]))
        for name in names
    }


def _traces(
    trajectory: Trajectory, names: List[str], counts: Dict[str, int]
) -> Dict[str, Any]:
    """Return bounded membrane/current/spike traces for each stage."""
    return {
        name: {
            "neurons": counts[name],
            "membrane": _rows(trajectory.membranes[name], MAX_NEURONS),
            "current": _rows(trajectory.currents[name], MAX_NEURONS),
            "spikes": _rows(trajectory.spikes[name], MAX_NEURONS),
        }
        for name in names
    }


def _educational(
    net: Any, spec: TopologySpec, spikes: torch.Tensor
) -> Trajectory:
    """Run the topology in educational mode, recording every trace."""
    shaped = input_shape.to_input_shape(spikes, spec)
    return run(net, shaped, mode=ExecutionMode.EDUCATIONAL)


def trajectory_payload(
    net: Any, spec: TopologySpec, spikes: torch.Tensor
) -> Dict[str, Any]:
    """Return bounded U[t]/I[t]/S[t] traces for the active model run."""
    trajectory = _educational(net, spec, spikes)
    names = _stage_names(trajectory)
    counts = _neuron_counts(trajectory, names)
    return {
        "steps": int(trajectory.steps),
        "stages": names,
        "caps": {"max_neurons": MAX_NEURONS, "max_stages": MAX_STAGES},
        "traces": _traces(trajectory, names, counts),
    }


def metrics_payload(
    net: Any, spec: TopologySpec, spikes: torch.Tensor
) -> Dict[str, Any]:
    """Return JSON-able trajectory metrics for the active model run."""
    return trajectory_metrics(_educational(net, spec, spikes))


def _benchmark_config(cfg: TrainConfig) -> BenchmarkConfig:
    """Return a tiny fixture, honouring an explicit topology choice."""
    if "topology" not in cfg.model_fields_set:
        return default_config()
    return replace(
        default_config(), topologies=(cfg.topology,), device=cfg.device
    )


def benchmark_payload(cfg: TrainConfig) -> Dict[str, Any]:
    """Return the JSON-able report of a tiny benchmark run."""
    return run_benchmark(_benchmark_config(cfg))
