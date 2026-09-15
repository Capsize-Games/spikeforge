"""The honest sparsity/drift report a ``prune`` call produces."""

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional


@dataclass(frozen=True)
class PruningReport:
    """What a pruning pass removed and how far the weights moved.

    ``sparsity`` is the achieved element sparsity and ``target_sparsity`` the
    request, so a granularity-induced mismatch is visible. ``drift`` holds the
    shared metric bundle for the flattened weights before and after.
    """

    strategy: str
    target_sparsity: float
    sparsity: float
    threshold: float
    pruned: int
    total: int
    tensors: Mapping[str, Any] = field(default_factory=dict)
    drift: Optional[Mapping[str, Any]] = field(default=None)

    def density(self) -> float:
        """Return the fraction of floating parameters that remain non-zero."""
        return 1.0 - self.sparsity

    def counts(self) -> Dict[str, int]:
        """Return the pruned and total element counts."""
        return {"pruned": self.pruned, "total": self.total}

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-serialisable form of the report."""
        return {
            "strategy": self.strategy,
            "target_sparsity": self.target_sparsity,
            "sparsity": self.sparsity,
            "density": self.density(),
            "threshold": self.threshold,
            "counts": self.counts(),
            "tensors": dict(self.tensors),
            "drift": None if self.drift is None else dict(self.drift),
        }
