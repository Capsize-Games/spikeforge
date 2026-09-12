"""``server.concurrency``: the server-wide training/pipeline job cap.

``TrainingService``/``PipelineService`` already refuse a second run within
one session; these tests cover the cross-session cap that sits above that,
plus the "server busy" error the WebSocket handlers surface when it is
full -- verified by monkeypatching the service's own ``start`` (a real run
would need a cached dataset and a background thread race to hit the cap
deterministically).
"""

import asyncio
from typing import Any, Dict, List

import pytest

from server import concurrency
from server.concurrency import (
    DEFAULT_MAX_CONCURRENT_JOBS,
    MAX_CONCURRENT_JOBS_ENV,
    SERVER_BUSY_MESSAGE,
)
from server.handlers import handle_train
from server.pipeline_handlers import handle_run_pipeline
from server.schemas import ClientMessage, PipelineGraphConfig, TrainConfig
from server.session import Session


class FakeWS:
    """Capture outbound messages instead of writing to a socket."""

    def __init__(self) -> None:
        """Start with an empty send log."""
        self.sent: List[Dict[str, Any]] = []

    async def send_json(self, message: Dict[str, Any]) -> None:
        """Record one outbound message."""
        self.sent.append(message)


@pytest.fixture(autouse=True)
def _reset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear the job-cap env var and the shared active-job counter."""
    monkeypatch.delenv(MAX_CONCURRENT_JOBS_ENV, raising=False)
    concurrency._active = 0  # noqa: SLF001 -- resetting shared test state


def test_max_concurrent_jobs_defaults_when_unset() -> None:
    """No env var falls back to the built-in default."""
    assert concurrency.max_concurrent_jobs() == DEFAULT_MAX_CONCURRENT_JOBS


def test_max_concurrent_jobs_reads_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid positive override is honoured."""
    monkeypatch.setenv(MAX_CONCURRENT_JOBS_ENV, "5")
    assert concurrency.max_concurrent_jobs() == 5


def test_max_concurrent_jobs_ignores_garbage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-integer or non-positive override falls back to the default."""
    monkeypatch.setenv(MAX_CONCURRENT_JOBS_ENV, "not-a-number")
    assert concurrency.max_concurrent_jobs() == DEFAULT_MAX_CONCURRENT_JOBS
    monkeypatch.setenv(MAX_CONCURRENT_JOBS_ENV, "0")
    assert concurrency.max_concurrent_jobs() == DEFAULT_MAX_CONCURRENT_JOBS


def test_try_acquire_refuses_past_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acquiring past the cap fails without changing the active count."""
    monkeypatch.setenv(MAX_CONCURRENT_JOBS_ENV, "2")
    assert concurrency.try_acquire() is True
    assert concurrency.try_acquire() is True
    assert concurrency.try_acquire() is False


def test_release_frees_a_slot_for_the_next_acquire(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Releasing a held slot lets a refused acquire succeed."""
    monkeypatch.setenv(MAX_CONCURRENT_JOBS_ENV, "1")
    assert concurrency.try_acquire() is True
    assert concurrency.try_acquire() is False
    concurrency.release()
    assert concurrency.try_acquire() is True


def test_release_never_goes_negative() -> None:
    """An extra release doesn't let the counter (and thus the cap) invert."""
    concurrency.release()
    concurrency.release()
    assert concurrency._active == 0  # noqa: SLF001 -- asserting test state


def test_handle_train_reports_busy_when_the_cap_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full job cap surfaces a clear error, not a silent no-op."""
    cfg = TrainConfig(
        dataset="mnist", topology="fc_small",
        topology_params={"hidden": 4, "num_classes": 3}, num_steps=2,
    )

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        monkeypatch.setattr(
            session.training, "start", lambda *a, **k: False
        )
        ws = FakeWS()
        await handle_train(ws, session, cfg)
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == SERVER_BUSY_MESSAGE


async def _drain_train_state(session: Session) -> None:
    """Consume the training queue until a stopped ``train_state`` lands."""
    while True:
        message = await asyncio.wait_for(session.training.queue.get(), 30)
        payload = message.get("payload")
        if (
            message["type"] == "train_state"
            and isinstance(payload, dict)
            and payload.get("running") is False
        ):
            return


def test_a_real_training_run_releases_its_slot_when_it_finishes() -> None:
    """The job cap doesn't leak: a finished run frees its slot for reuse."""
    cfg = TrainConfig(
        dataset="mnist", topology="fc_small",
        topology_params={"hidden": 4, "num_classes": 3},
        num_steps=2, subset=2, epochs=1, device="cpu",
    )

    async def _run() -> None:
        session = Session(asyncio.get_running_loop())
        ws = FakeWS()
        await handle_train(ws, session, cfg)
        await _drain_train_state(session)
        # The worker thread queues train_state just before returning, so
        # the outer finally's concurrency.release() can lag it by a beat.
        for _ in range(100):
            if concurrency._active == 0:  # noqa: SLF001
                return
            await asyncio.sleep(0.01)
        raise AssertionError("job slot was never released")

    asyncio.run(_run())


def test_handle_run_pipeline_reports_busy_when_the_cap_is_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same clear error on the pipeline-run path."""
    message = ClientMessage(
        type="run_pipeline",
        pipeline=PipelineGraphConfig(name="g", nodes=[]),
        pipeline_input={"frames": [], "encoded": False},
    )

    async def _run() -> List[Dict[str, Any]]:
        session = Session(asyncio.get_running_loop())
        monkeypatch.setattr(
            session.pipeline, "start", lambda *a, **k: False
        )
        ws = FakeWS()
        await handle_run_pipeline(ws, session, message)
        return ws.sent

    sent = asyncio.run(_run())
    assert [m["type"] for m in sent] == ["error"]
    assert sent[0]["payload"] == SERVER_BUSY_MESSAGE
