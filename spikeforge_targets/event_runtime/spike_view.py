"""Sparse views of binary spike frames and reusable density helpers."""

from dataclasses import dataclass
from typing import Tuple

import torch

from spikeforge.simulator import input_shape
from spikeforge.topology.spec import TopologySpec

#: Registry datasets are 1x28x28, so flat fixtures reshape the same way.
FLAT_FEATURES = 28 * 28
#: Default spike probability for a reusable low-density fixture.
SPIKE_PROBABILITY = 0.1


@dataclass(frozen=True)
class SparseSpikes:
    """A frame's nonzero entries as a COO tensor with density helpers."""

    sparse: torch.Tensor

    @classmethod
    def from_dense(cls, frame: torch.Tensor) -> "SparseSpikes":
        """Return the sparse COO view of ``frame`` (explicit zeros drop)."""
        return cls(frame.to_sparse())

    @property
    def indices(self) -> torch.Tensor:
        """Return the COO index tensor."""
        return self.sparse.indices()

    @property
    def values(self) -> torch.Tensor:
        """Return the COO value tensor."""
        return self.sparse.values()

    @property
    def size(self) -> Tuple[int, ...]:
        """Return the dense shape of the viewed frame."""
        return tuple(self.sparse.shape)

    @property
    def nnz(self) -> int:
        """Return the number of stored (nonzero) entries."""
        return int(self.values.numel())

    @property
    def numel(self) -> int:
        """Return the dense element count of the viewed frame."""
        return int(torch.Size(self.size).numel())

    @property
    def density(self) -> float:
        """Return the nonzero fraction in ``[0, 1]`` (0 for an empty frame)."""
        total = self.numel
        return 0.0 if total == 0 else self.nnz / total

    def torch_sparse(self) -> torch.Tensor:
        """Return the underlying COO tensor for a sparse matmul."""
        return self.sparse


def synthetic_spikes(
    spec: TopologySpec,
    steps: int,
    batch: int,
    seed: int,
    probability: float = SPIKE_PROBABILITY,
) -> torch.Tensor:
    """Return a seeded, low-density binary spike train shaped for ``spec``."""
    generator = torch.Generator().manual_seed(seed)
    flat = torch.rand(steps, batch, FLAT_FEATURES, generator=generator)
    shaped = input_shape.to_input_shape(flat, spec)
    return (shaped < probability).to(torch.float32)
