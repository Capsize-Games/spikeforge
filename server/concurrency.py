"""Server-wide cap on concurrently running training/pipeline jobs.

``TrainingService`` and ``PipelineService`` each already refuse a second run
*within one session* (``if session.training.is_running: return``), but
nothing stopped many WebSocket sessions from each starting their own job at
once and hanging the shared server. This adds one counter shared across
every ``Session`` instance in the process.
"""

import os
import threading

#: Environment variable overriding the default job cap.
MAX_CONCURRENT_JOBS_ENV = "SPIKEFORGE_DASHBOARD_MAX_CONCURRENT_JOBS"

#: Default number of training-or-pipeline jobs allowed to run at once,
#: server-wide, when the env var above is unset.
DEFAULT_MAX_CONCURRENT_JOBS = 2

_lock = threading.Lock()
_active = 0

#: Shared error text for a training/pipeline start refused by the job cap.
SERVER_BUSY_MESSAGE = (
    "server busy: too many training/pipeline jobs already running; "
    "try again shortly"
)


def max_concurrent_jobs() -> int:
    """Return the configured job cap, falling back to the default."""
    raw = os.environ.get(MAX_CONCURRENT_JOBS_ENV)
    if not raw:
        return DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT_JOBS
    return value if value > 0 else DEFAULT_MAX_CONCURRENT_JOBS


def try_acquire() -> bool:
    """Claim one job slot; False when the server is already at capacity."""
    global _active
    with _lock:
        if _active >= max_concurrent_jobs():
            return False
        _active += 1
        return True


def release() -> None:
    """Free one job slot."""
    global _active
    with _lock:
        _active = max(0, _active - 1)
