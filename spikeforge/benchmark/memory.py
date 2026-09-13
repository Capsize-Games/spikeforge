"""Best-effort memory probes for CPU and CUDA benchmark runs.

Every probe returns ``None`` when its metric is unavailable on this platform,
so a benchmark degrades to skipping rather than failing.
"""

import tracemalloc
from typing import Any, Callable, Dict, Optional

try:
    import resource
except ImportError:  # Windows does not provide the POSIX resource module.
    resource = None

import torch


def reset_cuda_peak(device: Any) -> None:
    """Reset CUDA peak-memory accounting when a GPU is in use."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def cuda_peak_bytes(device: Any) -> Optional[int]:
    """Return the CUDA peak allocated bytes on ``device``, or None on CPU."""
    if torch.device(device).type != "cuda":
        return None
    return int(torch.cuda.max_memory_allocated(device))


def process_rss_bytes() -> Optional[int]:
    """Return the process peak resident set size in bytes, or None."""
    if resource is None:
        return None
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
    except (ValueError, OSError):
        return None
    return int(usage.ru_maxrss) * 1024


def tracemalloc_peak_bytes(run_once: Callable[[], None]) -> Optional[int]:
    """Return the Python allocation peak in bytes during one ``run_once``.

    Returns None when ``tracemalloc`` cannot start; an error raised by
    ``run_once`` itself still propagates rather than being swallowed.
    """
    try:
        tracemalloc.start()
    except Exception:
        return None
    tracemalloc.reset_peak()
    try:
        run_once()
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    return int(peak)


def snapshot(run_once: Callable[[], None], device: Any) -> Dict[str, Any]:
    """Return peak CUDA, process RSS, and tracemalloc bytes for one run."""
    reset_cuda_peak(device)
    peak = tracemalloc_peak_bytes(run_once)
    return {
        "cuda_peak_bytes": cuda_peak_bytes(device),
        "process_rss_bytes": process_rss_bytes(),
        "tracemalloc_peak_bytes": peak,
    }
