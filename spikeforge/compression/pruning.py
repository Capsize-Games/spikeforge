"""Magnitude and structured pruning with an honest sparsity report.

Two strategies are offered:

``unstructured``  zero every element whose magnitude falls below a global
                  threshold chosen at the requested sparsity.
``structured``    zero whole output channels (dimension 0 groups) whose L2
                  norm falls below a global threshold chosen at the requested
                  fraction of groups.

Either way the *achieved* element sparsity is measured after pruning and
reported alongside the target, because ties and channel granularity mean the
two need not be equal. The induced drift against the original weights is a
pure metric from :mod:`spikeforge.nir_bridge.drift`, so a caller sees the
error rather than a promise.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from spikeforge.compression.errors import CompressionError

#: Names of the two supported pruning strategies.
STRATEGIES: Tuple[str, ...] = ("unstructured", "structured")


def _float_tensors(state_dict: Mapping[str, Any]) -> List[torch.Tensor]:
    """Return the floating tensors of ``state_dict`` in iteration order."""
    return [
        value
        for value in state_dict.values()
        if torch.is_tensor(value) and value.is_floating_point()
    ]


def sparsity(state_dict: Mapping[str, Any]) -> float:
    """Return the fraction of floating parameters that are exactly zero."""
    total = 0
    zeros = 0
    for tensor in _float_tensors(state_dict):
        total += int(tensor.numel())
        zeros += int((tensor == 0).sum())
    return zeros / total if total else 0.0


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


@dataclass(frozen=True)
class PrunedWeights:
    """A pruned state dict paired with the report describing it."""

    tensors: Mapping[str, torch.Tensor]
    report: PruningReport


def _require_sparsity(level: float) -> float:
    """Return a validated ``level`` in ``[0, 1)`` or raise a typed error."""
    value = float(level)
    if value < 0.0 or value >= 1.0:
        raise CompressionError(
            f"sparsity must be in [0, 1), got {level!r}"
        )
    return value


def _flatten(tensors: List[torch.Tensor]) -> torch.Tensor:
    """Return the concatenation of every tensor flattened to one dimension."""
    if not tensors:
        return torch.zeros(0)
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def _threshold(flat: torch.Tensor, level: float) -> float:
    """Return the magnitude threshold for the requested global sparsity."""
    if level <= 0.0 or not flat.numel():
        return 0.0
    return float(torch.quantile(flat.float().abs(), level))


def _magnitude_masks(
    state_dict: Mapping[str, Any], level: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Return a keep-mask per float tensor plus the global threshold."""
    flat = _flatten(_float_tensors(state_dict))
    cutoff = _threshold(flat, level)
    masks: Dict[str, torch.Tensor] = {}
    for name, value in state_dict.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        masks[name] = value.abs() > cutoff
    return masks, cutoff


def _structured_masks(
    state_dict: Mapping[str, Any], level: float
) -> Tuple[Dict[str, torch.Tensor], float]:
    """Return keep-masks that zero whole low-norm output channels.

    Only tensors with at least two dimensions are grouped, one L2 norm per
    leading-axis unit (a convolution's output channel, a linear layer's
    output unit); the groups are thresholded globally and the surviving units
    are broadcast back to the tensor's shape. One-dimensional tensors such as
    biases have no channel axis, so structured pruning leaves them untouched.
    """
    groups: List[Tuple[str, int, torch.Tensor]] = []
    norms: List[torch.Tensor] = []
    for name, value in state_dict.items():
        if not torch.is_tensor(value) or not value.is_floating_point():
            continue
        if value.dim() < 2:
            continue
        tensor = value.float()
        table = tensor.reshape(int(tensor.shape[0]), -1)
        norm = table.norm(dim=1)
        groups.append((name, int(tensor.shape[0]), norm))
        norms.append(norm)
    cutoff = _threshold(_flatten(norms), level)
    masks: Dict[str, torch.Tensor] = {}
    for name, row_count, norm in groups:
        spread = max(state_dict[name].dim() - 1, 0)
        keep = (norm > cutoff).reshape(row_count, *([1] * spread))
        masks[name] = keep.bool()
    return masks, cutoff


def _drift(
    before: Mapping[str, Any], after: Mapping[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return the flattened weight drift of ``after`` against ``before``."""
    from spikeforge.nir_bridge import drift

    left = _flatten(_float_tensors(before))
    right = _flatten(_float_tensors(after))
    if not left.numel():
        return None
    return drift.compare(right, left)


def _tensor_density(tensor: torch.Tensor) -> float:
    """Return the non-zero fraction of one tensor."""
    total = int(tensor.numel())
    if not total:
        return 1.0
    return 1.0 - int((tensor == 0).sum()) / total


def prune(
    state_dict: Mapping[str, Any],
    level: float,
    strategy: str = "unstructured",
) -> PrunedWeights:
    """Return ``state_dict`` pruned to ``level`` with its honest report.

    ``level`` is the requested element sparsity in ``[0, 1)``. An unknown
    ``strategy`` is refused with :class:`CompressionError`; the default
    ``unstructured`` zeroes by magnitude and ``structured`` zeroes by output
    channel. The original mapping is never mutated.
    """
    target = _require_sparsity(level)
    if strategy not in STRATEGIES:
        raise CompressionError(
            f"unknown pruning strategy {strategy!r}; expected one of "
            f"{STRATEGIES}"
        )
    if strategy == "structured":
        masks, cutoff = _structured_masks(state_dict, target)
    else:
        masks, cutoff = _magnitude_masks(state_dict, target)
    pruned: Dict[str, torch.Tensor] = {}
    per_tensor: Dict[str, Any] = {}
    for name, value in state_dict.items():
        if name in masks:
            alive = (value * masks[name]).to(value.dtype)
            pruned[name] = alive
            per_tensor[name] = {
                "density": _tensor_density(alive),
                "shape": [int(size) for size in value.shape],
            }
        else:
            pruned[name] = value
    total = sum(int(t.numel()) for t in _float_tensors(state_dict))
    pruned_count = sum(int((t == 0).sum()) for t in _float_tensors(pruned))
    achieved = pruned_count / total if total else 0.0
    report = PruningReport(
        strategy=strategy,
        target_sparsity=target,
        sparsity=achieved,
        threshold=cutoff,
        pruned=pruned_count,
        total=total,
        tensors=per_tensor,
        drift=_drift(state_dict, pruned),
    )
    return PrunedWeights(pruned, report)
