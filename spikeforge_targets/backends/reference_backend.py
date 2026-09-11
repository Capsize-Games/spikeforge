"""The always-available in-process reference backend.

It wraps :class:`~spikeforge.nir_bridge.interpreter.NirInterpreter`, so
the reference target keeps behaving exactly as before while presenting the
same compile/run interface as the SDK-backed targets.
"""

from typing import Any

from spikeforge.nir_bridge.interpreter import NirInterpreter
from spikeforge_targets.backends.result import (
    STATUS_OK,
    BackendResult,
)


class ReferenceBackend:
    """Execute a NIR graph with the independent reference interpreter."""

    name = "reference"

    def available(self) -> bool:
        """Return True; the reference interpreter needs no SDK."""
        return True

    def compile(self, graph: Any, spec: Any) -> Any:
        """Return ``graph`` unchanged; the interpreter executes it directly."""
        return graph

    def run(self, compiled: Any, spikes: Any) -> BackendResult:
        """Execute ``compiled`` over ``spikes`` and return its traces."""
        result = NirInterpreter(compiled).run(spikes)
        return BackendResult(
            target=self.name,
            status=STATUS_OK,
            steps=int(result.steps),
            readout=result.readout,
            spikes=result.spikes,
            membranes=result.membranes,
        )
