"""Serving-mode benchmark: p50/p99 latency, throughput, cold start, memory.

PT-W6 extends :mod:`spikeforge.benchmark` past the training-step harness with
an *in-process* serving benchmark over a :class:`.DeploymentBundle`. It drives
:class:`~spikeforge.serving.session.InferenceSession` predict/stream calls,
reports percentile latency and throughput at a chosen concurrency, and emits a
record shaped like the training harness (:func:`_record`) so the existing
:class:`~spikeforge.benchmark.store.BenchmarkStore` and the regression gate in
:mod:`spikeforge.benchmark.compare` consume it unchanged.

The report is plain JSON-able data and is honest about what is unavailable:
a percentile over zero samples is ``0.0`` and an unavailable throughput is
``None`` rather than a fabricated number.
"""

import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from spikeforge.benchmark import memory
from spikeforge.benchmark.store import BenchmarkStore
from spikeforge.benchmark.suite import with_metadata
from spikeforge.runtime import device as device_mod
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.session import InferenceSession

#: Mode label the serving records carry; the gate matches on it.
MODE = "serving"
#: Default raw sample geometry when the bundle pins no ``input_size``.
SAMPLE_SIZE: Tuple[int, int] = (28, 28)


@dataclass(frozen=True)
class ServingBenchmarkConfig:
    """A serving fixture over one ``.spkf`` bundle.

    ``calls`` predict calls are issued with ``concurrency`` worker sessions,
    each call advancing the session through one encoded sample. ``warmup``
    untimed calls precede the timed ones, and ``seed`` fixes the raw sample.
    """

    bundle: str
    calls: int = 8
    concurrency: int = 1
    warmup: int = 1
    seed: int = 0
    device: str = "cpu"


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Return the ``fraction`` (0..100) percentile of ``samples``."""
    ordered = sorted(float(sample) for sample in samples)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    rank = (float(fraction) / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _sample(bundle: DeploymentBundle, seed: int) -> torch.Tensor:
    """Return one seeded raw sample shaped for the bundle's geometry."""
    geometry = bundle.encode_spec().input_size or SAMPLE_SIZE
    height, width = (int(geometry[0]), int(geometry[1]))
    generator = torch.Generator().manual_seed(int(seed))
    return torch.rand(1, height, width, generator=generator)


def _one_call(session: InferenceSession, sample: torch.Tensor) -> int:
    """Drive one predict call (encode + every step) and return its steps."""
    steps = 0
    for _ in session.run_stream(session.encode(sample)):
        steps += 1
    return steps


def _timed_call(session: InferenceSession, sample: torch.Tensor) -> float:
    """Return the wall seconds one predict call took."""
    start = time.perf_counter()
    _one_call(session, sample)
    return time.perf_counter() - start


def _prepare(
    config: ServingBenchmarkConfig, device: torch.device
) -> Tuple[DeploymentBundle, InferenceSession, torch.Tensor, int, float]:
    """Load the bundle/session and time the cold start through one step."""
    start = time.perf_counter()
    bundle = DeploymentBundle.load(config.bundle, strict=True)
    session = InferenceSession.load(bundle, device=device)
    sample = _sample(bundle, config.seed)
    session.step_sample(sample)
    cold_start = time.perf_counter() - start
    steps = int(session.encode(sample).size(0))
    return bundle, session, sample, steps, cold_start


def _warm(session: InferenceSession, sample: torch.Tensor, runs: int) -> None:
    """Run ``runs`` untimed predict calls on ``session``."""
    for _ in range(max(int(runs), 0)):
        _one_call(session, sample)


def _sequential(
    session: InferenceSession, sample: torch.Tensor, calls: int
) -> Tuple[List[float], float]:
    """Time ``calls`` predict calls on one session, returning samples+wall."""
    durations: List[float] = []
    start = time.perf_counter()
    for _ in range(calls):
        durations.append(_timed_call(session, sample))
    return sorted(durations), time.perf_counter() - start


def _worker(
    session: InferenceSession,
    sample: torch.Tensor,
    calls: int,
    sink: List[float],
    lock: Any,
) -> None:
    """Time ``calls`` predict calls on ``session`` and merge into ``sink``."""
    local = [_timed_call(session, sample) for _ in range(calls)]
    with lock:
        sink.extend(local)


def _worker_call_counts(calls: int, concurrency: int) -> List[int]:
    """Return how many calls each of ``concurrency`` workers should run."""
    base, extra = divmod(calls, concurrency)
    return [base + (1 if i < extra else 0) for i in range(concurrency)]


def _concurrent(
    sessions: Sequence[InferenceSession], sample: torch.Tensor,
    calls: int, concurrency: int,
) -> Tuple[List[float], float]:
    """Time ``calls`` predict calls spread over ``concurrency`` workers."""
    per_worker = _worker_call_counts(calls, concurrency)
    sink: List[float] = []
    lock = threading.Lock()
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                _worker, sessions[index % len(sessions)], sample,
                per_worker[index], sink, lock,
            )
            for index in range(concurrency)
        ]
        for future in futures:
            future.result()
    return sorted(sink), time.perf_counter() - start


def _timed_run(
    config: ServingBenchmarkConfig, bundle: DeploymentBundle,
    session: InferenceSession, sample: torch.Tensor, device: torch.device,
) -> Tuple[List[float], float]:
    """Run the warmup and timed calls, returning latency samples+wall."""
    _warm(session, sample, config.warmup)
    calls = max(int(config.calls), 1)
    concurrency = max(int(config.concurrency), 1)
    if concurrency == 1:
        return _sequential(session, sample, calls)
    extra = [
        InferenceSession.load(bundle, device=device)
        for _ in range(concurrency - 1)
    ]
    for sibling in extra:
        _warm(sibling, sample, config.warmup)
    sessions = [session, *extra]
    return _concurrent(sessions, sample, calls, concurrency)


def _step_latencies(
    session: InferenceSession, sample: torch.Tensor
) -> List[float]:
    """Return the per-step wall seconds for one encoded sample."""
    spikes = session.encode(sample)
    durations: List[float] = []
    for index in range(int(spikes.size(0))):
        start = time.perf_counter()
        session.step(spikes[index])
        durations.append(time.perf_counter() - start)
    return durations


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


def run_serving_benchmark(
    config: ServingBenchmarkConfig,
) -> Dict[str, Any]:
    """Measure a bundle's in-process predict path, returning JSON-able data."""
    device = device_mod.resolve(config.device)
    bundle, session, sample, steps, cold_start = _prepare(config, device)
    durations, wall = _timed_run(config, bundle, session, sample, device)
    memory_info = memory.snapshot(lambda: _one_call(session, sample), device)
    step_durations = _step_latencies(session, sample)
    record = _record(
        config, bundle, durations, wall, steps, cold_start, memory_info,
        step_durations,
    )
    return {
        "config": asdict(config),
        "environment": _environment(device),
        "results": [record],
    }


def run_serving_suite(
    config: ServingBenchmarkConfig,
    store: Optional[BenchmarkStore] = None,
    label: Optional[str] = "serving",
    save: bool = True,
) -> Dict[str, Any]:
    """Benchmark one serving bundle and optionally persist the record."""
    report = run_serving_benchmark(config)
    enriched = with_metadata(report)
    if save:
        target = store if store is not None else BenchmarkStore()
        enriched["run_id"] = target.save(enriched, label=label)
    return enriched
