"""The backend-neutral inference interface behind ``InferenceSession``.

Every serving path used to execute through PyTorch with no alternative:
a session built a torch ``StageModule`` and stepped it under ``no_grad``.
This module defines the seam an alternative backend implements —
:func:`resolve_backend` validates the requested name, and
:class:`InferenceBackend` is the per-step contract a backend honors.
The torch ``StageModule`` path stays the default backend; the
``nir-c`` backend executes the same bundle through the
``nir-c-runtime`` C library instead.
"""

from typing import Any, Dict, Mapping, Protocol, Tuple

import torch

from spikeforge.serving.errors import BackendError
from spikeforge.topology.spec import TopologySpec

#: The default backend: the torch ``StageModule`` stepping path.
TORCH_BACKEND = "torch"
#: The C backend: fixed-timestep inference via ``nir-c-runtime``.
NIR_C_BACKEND = "nir-c"

#: Every backend name :func:`resolve_backend` accepts.
BACKENDS: Tuple[str, ...] = (TORCH_BACKEND, NIR_C_BACKEND)


def resolve_backend(name: str) -> str:
    """Return ``name`` when it names a backend, else raise.

    An unknown name is refused with :class:`BackendError` naming the
    accepted backends, before any bundle is loaded or module built.
    """
    if name in BACKENDS:
        return name
    raise BackendError(
        f"unknown inference backend {name!r}: "
        f"expected one of {', '.join(BACKENDS)}"
    )


class InferenceBackend(Protocol):
    """One timestep at a time, carrying whatever state the backend needs.

    A backend owns the per-step execution and its own carried state; the
    session owns the cumulative readout totals, the step count, and the
    :class:`~spikeforge.serving.prediction.Prediction` assembly, so every
    backend reports identical totals for identical per-step readouts.
    """

    @property
    def spec(self) -> TopologySpec:
        """Return the topology this backend executes."""
        ...  # pragma: no cover - protocol shape only

    def reset(self) -> None:
        """Clear the backend's carried temporal state."""
        ...  # pragma: no cover - protocol shape only

    def step(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one timestep; return ``(readout, spikes)``.

        ``readout`` is this step's output-stage value with its batch
        dimension; ``spikes`` maps neuron-stage names to this step's
        spikes (empty when the backend cannot observe them).
        """
        ...  # pragma: no cover - protocol shape only

    def capture_state(self) -> Dict[str, Any]:
        """Return the backend's serializable carried state."""
        ...  # pragma: no cover - protocol shape only

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`capture_state`."""
        ...  # pragma: no cover - protocol shape only

    def close(self) -> None:
        """Release any native resources; safe to call more than once."""
        ...  # pragma: no cover - protocol shape only
