"""Execute a target-ready graph through the Lava / Loihi 2 path.

The Lava backend follows the same protocol as the Norse simulator, but it is
the hardware path: when ``lava-nc`` is absent it reports ``unavailable`` with
a named reason, and when it is present it lowers a linear chain onto Lava
processes and runs them on an attached Loihi 2 when one has been opted into,
otherwise on the Loihi 2 CPU emulator. The result's ``path`` and notes always
name which path ran, so a simulation is never reported as a device
measurement.

The lowering is deliberately narrow: only a dense linear/neuron chain is
executable, because convolution and pooling have no faithful Lava process in
the supported surface. Anything else is refused with the offending node and
kind named rather than silently approximated.
"""

from math import exp
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from spikeforge_targets.backends import api, lowering
from spikeforge_targets.backends.errors import BackendUnavailableError
from spikeforge_targets.backends.result import (
    STATUS_OK,
    BackendResult,
)

#: Node kinds the Lava lowering can express as Lava processes.
LINEAR_KINDS = ("Affine", "Linear")
#: Note for a run on the Lava Loihi 2 CPU emulator.
EMULATOR_NOTE = (
    "ran on the Lava Loihi 2 CPU emulator; no physical device was probed"
)
#: Note for a run on an opted-in attached device.
DEVICE_NOTE = "ran on the Loihi 2 device opted in via SPIKEFORGE_LAVA_DEVICE"


def _linear_layer(node: Any) -> Dict[str, Any]:
    """Return a Dense layer description from a linear node."""
    weight = np.asarray(node.weight, dtype=np.float32)
    return {"kind": "Linear", "params": {"weight": weight},
            "size": int(weight.shape[0])}


def _neuron_layer(
    kind: str, node: Any, width: Optional[int]
) -> Dict[str, Any]:
    """Return a LIF/LI layer description at the tracked chain width."""
    if width is None:
        raise ValueError(f"lava lowering cannot size a leading {kind!r} node")
    decay = exp(-1.0 / lowering.scalar(node.tau))
    return {
        "kind": kind,
        "params": {
            "decay": 1.0 - decay,
            "v_threshold": lowering.scalar(node.v_threshold),
        },
        "size": width,
    }


def lower_program(graph: Any) -> Dict[str, Any]:
    """Lower a linear graph to a Lava layer program, or raise a reason."""
    layers: List[Dict[str, Any]] = []
    width: Optional[int] = None
    for name, node in lowering.linear_chain(graph):
        kind = type(node).__name__
        if kind in ("Input", "Output"):
            continue
        if kind in LINEAR_KINDS:
            layer = _linear_layer(node)
            width = layer["size"]
        elif kind in ("LIF", "LI"):
            layer = _neuron_layer(kind, node, width)
        else:
            raise ValueError(
                f"lava lowering does not support node {name!r} of kind "
                f"{kind!r}"
            )
        layers.append(layer)
    return {"layers": layers}


class LavaBackend:
    """Compile and run a linear NIR chain through Lava on Loihi 2."""

    name = "lava_loihi2"

    def available(self) -> bool:
        """Return True when the ``lava`` package can be imported."""
        return api.module_available("lava")

    def compile(self, graph: Any, spec: Any) -> Any:
        """Lower ``graph`` to a Lava program, or raise a named reason."""
        if not self.available():
            raise BackendUnavailableError(self.name, "lava")
        return lower_program(graph)

    def _result(self, program: Any, spikes: Any) -> BackendResult:
        """Run ``program`` on Lava and shape the result honestly."""
        device = api.lava_device_present()
        raw = api.lava_run(program, spikes, device)
        readout = torch.as_tensor(
            np.asarray(raw["readout"]),
            dtype=spikes.dtype,
            device=spikes.device,
        )
        note = DEVICE_NOTE if device else EMULATOR_NOTE
        path = raw.get("path", "loihi2_device" if device else "loihi2_sim")
        return BackendResult(
            target=self.name,
            status=STATUS_OK,
            steps=int(spikes.size(0)),
            readout=readout,
            notes=(note,),
            path=path,
        )

    def run(self, compiled: Any, spikes: Any) -> BackendResult:
        """Execute ``compiled`` on Lava and return the path-named result."""
        return self._result(compiled, spikes)
