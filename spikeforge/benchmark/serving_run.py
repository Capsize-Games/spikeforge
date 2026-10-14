"""Drive timed predict calls over bundle sessions, sequentially or not."""

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Sequence, Tuple

import torch

from spikeforge.benchmark.serving_config import (
    SAMPLE_SIZE,
    ServingBenchmarkConfig,
)
from spikeforge.serving.bundle import DeploymentBundle
from spikeforge.serving.session import InferenceSession


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
    """Load the bundle/session and time cold start through one sample."""
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
