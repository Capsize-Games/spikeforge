"""The default inference backend: the torch ``StageModule`` path.

This is the stepping the session has always done — a torch module
advanced one timestep at a time under ``no_grad`` through the shared
:func:`~spikeforge.simulator.step_stages.step_stages` body — behind
the :class:`~spikeforge.serving.inference_backend.InferenceBackend`
seam, so the session can swap it for the C runtime without changing
what this path computes.
"""

from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch

from spikeforge.serving.errors import BackendError, StateError
from spikeforge.serving.state_tree import StateTree
from spikeforge.simulator.grad_policy import detach_state
from spikeforge.simulator.module_spec import spec_of
from spikeforge.simulator.state import initial_state, neuron_kinds
from spikeforge.simulator.step_stages import PostStep, step_stages
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import (
    CURRENT_KEY,
    PREV_KEY,
    StageModule,
)

#: Carried-state entries that are not neuron stages.
_RESERVED: Tuple[str, ...] = (PREV_KEY, CURRENT_KEY)


class TorchBackend:
    """Step a torch module, carrying its neuron state between steps."""

    def __init__(
        self,
        module: StageModule,
        device: Union[str, torch.device] = "cpu",
        post_step: Optional[PostStep] = None,
    ) -> None:
        """Wrap ``module``; see the session's ``post_step`` property."""
        self._module = module
        self._device = torch.device(device)
        self._post_step = post_step
        self._spec: TopologySpec = spec_of(module)
        self._neurons: Tuple[str, ...] = tuple(neuron_kinds(self._spec))
        self._state: Optional[Dict[str, Any]] = None
        self._closed = False

    @property
    def spec(self) -> TopologySpec:
        """Return the topology this backend executes."""
        return self._spec

    @property
    def post_step(self) -> Optional[PostStep]:
        """Return the per-step transform, or ``None`` for plain steps."""
        return self._post_step

    def reset(self) -> None:
        """Clear the carried neuron state."""
        self._require_open("reset")
        self._state = None

    def step(
        self, frame: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Advance one timestep; return ``(readout, spikes)``."""
        self._require_open("step")
        with torch.no_grad():
            if self._state is None:
                self._state = dict(initial_state(self._spec, frame))
            outputs, state = step_stages(
                self._module, frame, self._state,
                post_step=self._post_step,
            )
            readout = outputs[self._spec.output].detach()
            self._state = detach_state(state)
            return readout, self._spikes(outputs)

    def capture_state(self) -> Dict[str, Any]:
        """Return the backend's serializable carried state."""
        return StateTree(self._state).to_dict()

    def restore_state(self, state: Mapping[str, Any]) -> None:
        """Restore a state produced by :meth:`capture_state`.

        A state whose stage names do not match this backend's spec is
        refused with :class:`StateError`.
        """
        tree = StateTree.from_dict(dict(state)).to(self._device)
        self._validate(tree.raw())
        self._state = dict(tree.raw()) or None

    def close(self) -> None:
        """Close the backend; stepping or resetting after this refuses.

        Torch owns no native resources to release, so closing only
        retires the backend — matching the C backend's terminal close.
        """
        self._closed = True

    def _require_open(self, verb: str) -> None:
        """Raise when the backend was closed before ``verb``."""
        if self._closed:
            raise BackendError(f"this torch backend is closed: {verb} refused")

    def _spikes(
        self, outputs: Mapping[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Return this step's neuron-stage spikes, bounded to the neurons."""
        return {
            name: outputs[name].detach()
            for name in self._neurons
            if name in outputs
        }

    def _validate(self, raw: Mapping[str, Any]) -> None:
        """Raise when ``raw`` names stages this backend's spec lacks."""
        if not raw:
            return
        present = set(raw) - set(_RESERVED)
        unknown = present - set(self._neurons)
        missing = set(self._neurons) - present
        if unknown or missing:
            raise StateError(
                "stage mismatch: unknown="
                + repr(sorted(unknown))
                + " missing="
                + repr(sorted(missing))
            )
