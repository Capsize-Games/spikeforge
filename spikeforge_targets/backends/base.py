"""The structural type every executable backend implements."""

from typing import Any, Protocol

from spikeforge_targets.backends.result import BackendResult


class Backend(Protocol):
    """Compile and run a target-ready NIR graph on one backend."""

    name: str

    def available(self) -> bool:
        """Return True when the backend's SDK is importable here."""
        ...

    def compile(self, graph: Any, spec: Any) -> Any:
        """Lower ``graph`` to a runnable program or raise a named error."""
        ...

    def run(self, compiled: Any, spikes: Any) -> BackendResult:
        """Execute ``compiled`` over a ``[T, ...]`` spike train."""
        ...
