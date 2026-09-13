"""LIF readout over a Hebbian synapse: one-shot, non-forgetting memory."""

from typing import Optional

import snntorch as snn
import torch

from spikeforge.memory.hebbian_synapse import HebbianSynapse

#: Default membrane decay for the memory neurons.
DEFAULT_BETA = 0.9
#: Default firing threshold for the memory neurons.
DEFAULT_THRESHOLD = 1.0
#: One-shot write gain: scales a normalised trace onto a fresh column.
DEFAULT_TEACH_GAIN = 4.0


class OneShotAssociativeMemory:
    """One LIF neuron per taught class, bound in a single Hebbian write.

    Sits beside a frozen classifier and reads its hidden-layer spikes.
    :meth:`teach` performs exactly one weight write per new class and
    never touches an existing class's column, so retention of earlier
    classes is a structural guarantee rather than a measured outcome.
    Recall (:meth:`step`) is ordinary ``snn.Leaky`` dynamics over the
    Hebbian synapse's current -- no classical nearest-neighbour lookup.
    """

    def __init__(
        self,
        in_features: int,
        beta: float = DEFAULT_BETA,
        threshold: float = DEFAULT_THRESHOLD,
        device: Optional[torch.device] = None,
    ) -> None:
        """Start with no taught classes; :meth:`teach` adds them."""
        self._device = device or torch.device("cpu")
        self._synapse = HebbianSynapse(in_features, self._device)
        self._neuron = snn.Leaky(
            beta=beta, threshold=threshold, reset_mechanism="subtract",
        ).to(self._device)
        self._mem: Optional[torch.Tensor] = None

    @property
    def num_classes(self) -> int:
        """Return how many classes have been taught so far."""
        return self._synapse.out_features

    def reset_state(self, batch_size: int = 1) -> None:
        """Start a fresh membrane state for the next temporal run."""
        self._mem = torch.zeros(
            batch_size, self.num_classes, device=self._device,
        )

    def step(self, hidden_spikes: torch.Tensor) -> torch.Tensor:
        """Advance one step given the base net's hidden-layer spikes.

        ``hidden_spikes`` is ``[B, H]``; returns ``[B, num_classes]``
        memory-neuron spikes for this step.
        """
        batch_size = hidden_spikes.size(0)
        if self._mem is None or self._mem.size(0) != batch_size:
            self.reset_state(batch_size)
        current = self._synapse.forward(hidden_spikes)
        spikes, self._mem = self._neuron(current, self._mem)
        return spikes

    def teach(
        self,
        hidden_spike_train: torch.Tensor,
        gain: float = DEFAULT_TEACH_GAIN,
    ) -> int:
        """One-shot bind a single example's spikes to a new class.

        ``hidden_spike_train`` is ``[T, H]`` for one example. Returns
        the new class's index, stable for the module's lifetime and
        mapped back to a real label by the caller.
        """
        return self.teach_many(hidden_spike_train.unsqueeze(0), gain)

    def teach_many(
        self,
        hidden_spike_trains: torch.Tensor,
        gain: float = DEFAULT_TEACH_GAIN,
    ) -> int:
        """Bind a class from several examples with one averaged write.

        ``hidden_spike_trains`` is ``[N, T, H]``. Averaging the examples'
        spike-count traces keeps this a single Hebbian write while reducing
        sensitivity to the particular exemplar used for teaching.
        """
        if hidden_spike_trains.ndim != 3:
            raise ValueError("teaching examples must have shape [N, T, H]")
        index = self._synapse.grow()
        trace = hidden_spike_trains.sum(dim=1).mean(dim=0)
        self._synapse.write(trace, index, gain)
        self._mem = None
        return index
