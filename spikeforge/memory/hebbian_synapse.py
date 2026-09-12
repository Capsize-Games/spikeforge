"""A feedforward synapse written by local Hebbian correlation."""

from typing import Optional

import torch


class HebbianSynapse:
    """Weights grown and written one column at a time, never by backprop.

    Every other neuron kind in :mod:`spikeforge.neurons` learns its
    weights through the optimizer in ``TrainingEngine``. This synapse
    is the deliberate exception: :meth:`grow` appends one zero output
    column per new class, and :meth:`write` is the only way a column
    is ever changed, via a single outer-product correlation between a
    presynaptic spike-count trace and that one column. No optimizer,
    no loss, no step touches any other column.
    """

    def __init__(
        self, in_features: int, device: Optional[torch.device] = None,
    ) -> None:
        """Start with zero output columns; :meth:`grow` adds one."""
        self._in_features = in_features
        self._device = device or torch.device("cpu")
        self._weight = torch.zeros(in_features, 0, device=self._device)

    @property
    def out_features(self) -> int:
        """Return how many classes have been taught so far."""
        return self._weight.size(1)

    def grow(self) -> int:
        """Append one zero-initialised output column; return its index."""
        column = torch.zeros(
            self._in_features, 1, device=self._device,
        )
        self._weight = torch.cat([self._weight, column], dim=1)
        return self.out_features - 1

    def write(
        self, trace: torch.Tensor, index: int, gain: float,
    ) -> None:
        """Hebbian outer-product write of ``trace`` into column ``index``.

        ``trace`` is the presynaptic activity for one taught example
        (e.g. summed spike counts over time), normalised to unit norm
        before being scaled by ``gain`` so the write's strength does
        not depend on how many spikes the example happened to emit.
        """
        norm = trace.norm().clamp_min(1e-8)
        with torch.no_grad():
            self._weight[:, index] += gain * trace.to(self._device) / norm

    def forward(self, spikes: torch.Tensor) -> torch.Tensor:
        """Return the synaptic current ``spikes @ weight`` for one step."""
        return spikes @ self._weight
