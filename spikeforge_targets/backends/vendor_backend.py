"""Execute a target-ready graph on a vendor SNN simulator.

Speck (Sinabs), Xylo (Rockpool), and SpiNNaker2 expose their software
simulators through separate SDKs. Each follows the same backend protocol as the
Norse and Lava backends: availability comes from the isolated probe in
:mod:`spikeforge_targets.backends.api` (module import plus a minimal capability
check), the graph is lowered with the shared linear lowering, and the run is
delegated to an isolated SDK touch-point that names the simulator path. A
missing SDK yields ``unavailable`` with a named reason; SDK drift raises and is
reported as a named ``error`` rather than a silent wrong answer.

No path claims a device measurement: every vendor result sets ``estimate:
true`` and names its simulator, so a Speck/Xylo/SpiNNaker2 run is never read as
an on-chip or on-board measurement.
"""

from typing import Any, Dict, Mapping

import torch

from spikeforge_targets.backends import api, lowering
from spikeforge_targets.backends.errors import BackendUnavailableError
from spikeforge_targets.backends.result import (
    STATUS_OK,
    BackendResult,
)

#: Note attached to every Speck run.
SPECK_NOTE = (
    "ran on the Sinabs Speck software simulator; no chip was probed"
)
#: Note attached to every Xylo run.
XYLO_NOTE = (
    "ran on the Rockpool Xylo software simulator; no chip was probed"
)
#: Note attached to every SpiNNaker2 run.
SPINNAKER2_NOTE = (
    "ran on the SpiNNaker2 host simulator; no board was probed"
)
#: The isolated SDK touch-point signature: ``(program, spikes) -> mapping``.
RunAttr = str


class VendorBackend:
    """Compile and run a linear NIR chain on one vendor SNN simulator.

    The SDK touch-point is looked up on :mod:`spikeforge_targets.backends.api`
    at run time (by attribute name), so an isolated stub, and the honest
    absence handling in that module, apply without rebinding the backend.
    """

    def __init__(
        self,
        name: str,
        extra: str,
        run_attr: RunAttr,
        path: str,
        note: str,
    ) -> None:
        """Record the target name, its pip extra, and its SDK touch-point."""
        self.name = name
        self.extra = extra
        self._run_attr = run_attr
        self._path = path
        self._note = note

    def available(self) -> bool:
        """Return True when the SDK imports and meets its capability check."""
        return api.backend_available(self.name)

    def compile(self, graph: Any, spec: Any) -> Any:
        """Lower ``graph`` to a linear layer program, or raise a reason."""
        if not self.available():
            raise BackendUnavailableError(self.name, self.extra)
        return lowering.linear_program(graph)

    def run(self, compiled: Any, spikes: Any) -> BackendResult:
        """Execute ``compiled`` on the vendor simulator and name the path."""
        run_fn = getattr(api, self._run_attr)
        raw: Mapping[str, Any] = run_fn(compiled, spikes)
        readout = torch.as_tensor(
            raw["readout"], dtype=spikes.dtype, device=spikes.device
        )
        return BackendResult(
            target=self.name,
            status=STATUS_OK,
            steps=int(spikes.size(0)),
            readout=readout,
            notes=(self._note,),
            path=str(raw.get("path", self._path)),
            estimate=True,
        )


def vendor_backends() -> Dict[str, VendorBackend]:
    """Return the vendor simulator backends keyed by target name."""
    return {
        "speck": VendorBackend(
            "speck", "speck", "sinabs_run", "speck_simulator", SPECK_NOTE
        ),
        "xylo": VendorBackend(
            "xylo", "xylo", "rockpool_run", "xylo_simulator", XYLO_NOTE
        ),
        "spinnaker2": VendorBackend(
            "spinnaker2",
            "spinnaker2",
            "spinnaker2_run",
            "spinnaker2_simulator",
            SPINNAKER2_NOTE,
        ),
    }
