"""Typed errors naming which verification step failed, and why."""


class VerificationStepError(Exception):
    """Raised by one pipeline stage with an actionable, named reason.

    ``step`` identifies which stage of the pipeline rejected the artifact
    (e.g. ``"fetch"``, ``"bundle"``, ``"drift"``) and ``reason`` is the
    human-readable detail shown to the uploader -- "weights are not
    loadable" rather than "verification failed" (the honesty bar
    ``docs/model-hub.md`` states for every rejection in this project).
    """

    def __init__(self, step: str, reason: str) -> None:
        """Record ``step`` and ``reason`` and build a clear message."""
        super().__init__(f"{step}: {reason}")
        self.step: str = step
        self.reason: str = reason


class FetchError(VerificationStepError):
    """Raised when the artifact cannot be fetched from its signed URL."""

    def __init__(self, reason: str) -> None:
        """Record the fetch failure's ``reason``."""
        super().__init__("fetch", reason)


class CallbackError(Exception):
    """Raised when the signed report cannot be posted back to the hub."""
