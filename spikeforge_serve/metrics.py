"""Serving-level instrumentation wired into the core observability registry.

PT-W6 asks for request-level counters and a latency histogram over the
existing :mod:`spikeforge.observability` registry, so the service never grows
a second metrics store. Every name is dotted and therefore sanitised by the
core Prometheus exporter (``serve.request_seconds`` becomes the histogram
``serve_request_seconds``).

Recorded series:

* ``serve.requests``       - counter of handled HTTP requests;
* ``serve.errors``         - counter of responses with status >= 400;
* ``serve.in_flight``      - gauge of requests currently being handled;
* ``serve.request_seconds``- histogram of request latency;
* ``serve.steps``          - counter of timesteps the sessions advanced;
* ``serve.stream_frames``  - counter of WebSocket stream frames processed.

The ``/metrics`` scrape itself is *not* instrumented so a scrape cannot
inflate the request counters it is reading.
"""

import threading
import time
from contextlib import contextmanager
from typing import Iterator

from spikeforge.observability import metrics, prometheus

#: Counter of handled HTTP requests.
REQUESTS = "serve.requests"
#: Counter of responses that failed (status >= 400).
ERRORS = "serve.errors"
#: Gauge of requests currently in flight.
IN_FLIGHT = "serve.in_flight"
#: Histogram of request latency, in seconds.
REQUEST_SECONDS = "serve.request_seconds"
#: Counter of timesteps advanced by the serving sessions.
STEPS = "serve.steps"
#: Counter of WebSocket frames processed by the stream route.
STREAM_FRAMES = "serve.stream_frames"

#: Path whose requests are excluded from the request counters.
METRICS_PATH = "/metrics"

#: Status at or above which a response counts as an error.
_ERROR_STATUS = 400


class _InFlight:
    """Track concurrent requests and mirror the count into a gauge."""

    def __init__(self) -> None:
        """Create a zeroed in-flight counter."""
        self._lock = threading.Lock()
        self._count = 0

    def enter(self) -> None:
        """Record a request starting and publish the new gauge value."""
        with self._lock:
            self._count += 1
            value = self._count
        metrics.gauge(IN_FLIGHT, value)

    def exit(self) -> None:
        """Record a request finishing and publish the new gauge value."""
        with self._lock:
            self._count = max(0, self._count - 1)
            value = self._count
        metrics.gauge(IN_FLIGHT, value)


_in_flight = _InFlight()


@contextmanager
def track_in_flight() -> Iterator[None]:
    """Count one request as in flight for the duration of the block."""
    _in_flight.enter()
    try:
        yield
    finally:
        _in_flight.exit()


def observe_request(status: int, seconds: float) -> None:
    """Record one finished request's status and latency."""
    metrics.counter(REQUESTS)
    metrics.observe(REQUEST_SECONDS, seconds)
    if int(status) >= _ERROR_STATUS:
        metrics.counter(ERRORS)


def count_steps(amount: int = 1) -> None:
    """Add ``amount`` to the timestep counter."""
    if amount:
        metrics.counter(STEPS, amount)


def count_stream_frames(amount: int = 1) -> None:
    """Add ``amount`` to the WebSocket frame counter."""
    if amount:
        metrics.counter(STREAM_FRAMES, amount)


def elapsed_since(start: float) -> float:
    """Return the seconds elapsed since a ``perf_counter`` reading."""
    return time.perf_counter() - start


def render_text() -> str:
    """Render the shared registry as Prometheus exposition text."""
    return prometheus.render(metrics.registry())
