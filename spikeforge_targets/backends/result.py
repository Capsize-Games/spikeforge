"""The run result of one backend execution.

``status`` is one of ``ok``, ``unavailable``, or ``error`` and is always
honest: an absent SDK yields ``unavailable`` with a named reason, a refused
graph or a raising SDK yields ``error`` with the reason in ``notes``, and
only a completed run yields ``ok``. ``readout`` and the per-node spike and
membrane traces mirror
:class:`~spikeforge.nir_bridge.interpreter_result.InterpreterResult`.
``path`` names which execution path ran (for example a Lava emulator versus
an attached device) so a caller never mistakes a simulation for a measurement.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

#: The status values a backend result can carry.
STATUS_OK = "ok"
STATUS_UNAVAILABLE = "unavailable"
STATUS_ERROR = "error"


@dataclass(frozen=True)
class BackendResult:
    """The honest outcome of compiling and running a graph on a backend."""

    target: str
    status: str
    steps: int = 0
    readout: Optional[torch.Tensor] = None
    spikes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    membranes: Mapping[str, torch.Tensor] = field(default_factory=dict)
    notes: Tuple[str, ...] = ()
    path: Optional[str] = None
    rewritten: Optional[Mapping[str, Any]] = None
    compare: Optional[Mapping[str, Any]] = None
    quantization: Optional[Mapping[str, Any]] = None

    def ok(self) -> bool:
        """Return True when the backend completed a run."""
        return self.status == STATUS_OK

    def _readout(self) -> List[float]:
        """Return the readout as a plain float list."""
        if self.readout is None:
            return []
        return [float(value) for value in self.readout.reshape(-1)]

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the result."""
        return {
            "target": self.target,
            "status": self.status,
            "steps": int(self.steps),
            "path": self.path,
            "readout": self._readout(),
            "spike_nodes": sorted(self.spikes),
            "membrane_nodes": sorted(self.membranes),
            "notes": list(self.notes),
            "rewritten": None if self.rewritten is None else dict(
                self.rewritten
            ),
            "compare": None if self.compare is None else dict(self.compare),
            "quantization": (
                None if self.quantization is None else dict(self.quantization)
            ),
        }
