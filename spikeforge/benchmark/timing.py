"""Wall-clock timing helpers for the benchmark harness."""

import statistics
import time
from typing import Any, Callable, Dict, List

import torch


def synchronize(device: Any) -> None:
    """Block until pending CUDA work on ``device`` completes."""
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def clock(device: Any) -> float:
    """Return a CUDA-synchronised performance-counter reading in seconds."""
    synchronize(device)
    return time.perf_counter()


def summarise(times: List[float], steps: int) -> Dict[str, Any]:
    """Turn per-pass wall times (seconds) into step-level statistics."""
    mean = statistics.fmean(times)
    median = statistics.median(times)
    mean_ms = mean * 1000.0
    median_ms = median * 1000.0
    return {
        "total_ms": mean_ms,
        "median_total_ms": median_ms,
        "mean_ms_per_step": mean_ms / steps,
        "median_ms_per_step": median_ms / steps,
        "steps_per_second": steps / mean if mean > 0 else None,
        "repeats": len(times),
    }


def time_calls(
    run_once: Callable[[], None],
    device: Any,
    repeats: int,
    warmup: int,
    steps: int,
) -> Dict[str, Any]:
    """Time ``repeats`` runs of ``run_once`` after ``warmup`` untimed runs."""
    for _ in range(max(int(warmup), 0)):
        run_once()
    samples: List[float] = []
    for _ in range(max(int(repeats), 1)):
        start = clock(device)
        run_once()
        samples.append(clock(device) - start)
    return summarise(samples, steps)
