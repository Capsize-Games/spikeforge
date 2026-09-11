"""Opt-in gradient policies for the shared temporal loop.

Both policies are strictly additive. Left at their defaults the loop behaves
exactly as before: no activations are recomputed and no gradient path is cut.

Gradient checkpointing recomputes each step's activations during the backward
pass instead of storing them, trading extra compute for a smaller activation
footprint on long ``T`` runs. Truncated backpropagation through time (streaming
BPTT) detaches the carried neuron state every ``bptt_steps`` steps, bounding
the gradient horizon; forward values are untouched.
"""

from typing import Any, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint

_StepResult = Tuple[Any, Any]


def _window(bptt_steps: Optional[int]) -> Optional[int]:
    """Normalise a BPTT window, treating non-positive values as disabled."""
    if bptt_steps is None:
        return None
    steps = int(bptt_steps)
    return steps if steps > 0 else None


def detach_state(value: Any) -> Any:
    """Return ``value`` with every nested tensor detached from the graph.

    Non-tensor leaves (e.g. ``None``) are returned unchanged, so the state
    mapping keeps its exact structure and only the gradient path is cut.
    """
    if isinstance(value, torch.Tensor):
        return value.detach()
    if isinstance(value, dict):
        return {key: detach_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_state(item) for item in value)
    return value


class GradPolicy:
    """Apply opt-in checkpointing and truncated BPTT to one run's steps."""

    def __init__(
        self,
        checkpointing: bool = False,
        bptt_steps: Optional[int] = None,
    ) -> None:
        """Store the default-off checkpointing and BPTT settings."""
        self._checkpointing = bool(checkpointing)
        self._bptt = _window(bptt_steps)

    @property
    def checkpointing(self) -> bool:
        """Return True when each step recomputes its activations."""
        return self._checkpointing

    @property
    def bptt_steps(self) -> Optional[int]:
        """Return the gradient-horizon window, or None for full BPTT."""
        return self._bptt

    def step(self, step_fn: Any, x: torch.Tensor, state: Any) -> _StepResult:
        """Run one step, recomputing its activations when opted in."""
        if not self._checkpointing:
            return step_fn(x, state)
        return checkpoint(step_fn, x, state, use_reentrant=False)

    def after_step(self, state: Any, index: int) -> Any:
        """Detach the carried state at each BPTT window boundary."""
        if self._bptt is None or (index + 1) % self._bptt:
            return state
        return detach_state(state)
