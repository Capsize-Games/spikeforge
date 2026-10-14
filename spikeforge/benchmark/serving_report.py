"""Assemble the JSON-ready serving result record and its environment."""

import statistics
from typing import Any, Dict, Optional, Sequence

import torch

from spikeforge.benchmark.serving_config import MODE, ServingBenchmarkConfig
from spikeforge.benchmark.serving_run import percentile
from spikeforge.serving.bundle import DeploymentBundle


def _forward_block(
    durations: Sequence[float], steps: int
) -> Dict[str, Any]:
    """Return the training-shaped ``forward`` block for a serving record."""
    mean = statistics.fmean(durations)
    median = statistics.median(durations)
    mean_ms = mean * 1000.0
    per_step = mean_ms / steps if steps else None
    return {
        "total_ms": mean_ms,
        "median_total_ms": median * 1000.0,
        "mean_ms_per_step": per_step,
        "median_ms_per_step": median * 1000.0 / steps if steps else None,
        "steps_per_second": steps / mean if mean > 0 else None,
        "repeats": len(durations),
    }


def _peak_memory(memory_info: Dict[str, Any]) -> Optional[int]:
    """Return the best available peak-memory reading."""
    return memory_info.get("process_rss_bytes") or memory_info.get(
        "tracemalloc_peak_bytes"
    )


def _serving_rates(
    durations: Sequence[float], wall: float, steps: int
) -> Dict[str, Any]:
    """Return the throughput/percentile fields of the serving block."""
    calls = len(durations)
    return {
        "calls": calls,
        "frames_per_call": steps,
        "p50_ms": percentile(durations, 50.0) * 1000.0,
        "p99_ms": percentile(durations, 99.0) * 1000.0,
        "mean_ms": statistics.fmean(durations) * 1000.0 if durations else 0.0,
        "throughput_per_second": calls / wall if wall > 0 else None,
        "steps_per_second": calls * steps / wall if wall > 0 else None,
    }


def _serving_block(
    config: ServingBenchmarkConfig,
    durations: Sequence[float],
    wall: float,
    steps: int,
    cold_start: float,
    memory_info: Dict[str, Any],
    step_durations: Sequence[float],
) -> Dict[str, Any]:
    """Return the serving-specific latency/throughput block."""
    fields = _serving_rates(durations, wall, steps)
    fields.update(
        concurrency=max(int(config.concurrency), 1),
        cold_start_ms=cold_start * 1000.0,
        peak_memory_bytes=_peak_memory(memory_info),
        step_p50_ms=percentile(step_durations, 50.0) * 1000.0,
        step_p99_ms=percentile(step_durations, 99.0) * 1000.0,
    )
    return fields


def _record(
    config: ServingBenchmarkConfig, bundle: DeploymentBundle,
    durations: Sequence[float], wall: float, steps: int,
    cold_start: float, memory_info: Dict[str, Any],
    step_durations: Sequence[float],
) -> Dict[str, Any]:
    """Assemble one JSON-ready serving result record."""
    return {
        "topology": str(bundle.manifest.get("topology") or "bundle"),
        "mode": MODE,
        "compiled": False,
        "compile_status": "eager",
        "forward": _forward_block(durations, steps),
        "backward": None,
        "memory": memory_info,
        "serving": _serving_block(
            config, durations, wall, steps, cold_start, memory_info,
            step_durations,
        ),
    }


def _environment(device: torch.device) -> Dict[str, Any]:
    """Describe the torch build and resolved device for the report."""
    return {
        "torch_version": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
