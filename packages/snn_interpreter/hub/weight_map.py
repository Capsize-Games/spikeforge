"""Load compatible artifact weights into a freshly built preset module.

Weights are mapped only for an exact structural match, and the load refuses
any strict mismatch: a missing, unexpected, or shape-mismatched key is reported
by name and nothing is applied. This is the difference between mapping a model
and silently loading the wrong tensors.

Two sources are supported. A torch ``state_dict`` maps by key directly. A NIR
graph maps by node name, taking each structural node's ``weight`` and ``bias``
fields (the only tensor-bearing node fields the exporter emits).
"""

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

#: Node fields carrying learned tensors in an exported graph.
_WEIGHT_FIELDS = ("weight", "bias")


@dataclass(frozen=True)
class WeightLoadResult:
    """Outcome of mapping artifact weights onto a built module."""

    loaded: bool
    missing: Tuple[str, ...]
    unexpected: Tuple[str, ...]
    mismatched: Tuple[str, ...]
    note: str

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able weight-load report."""
        return {
            "loaded": self.loaded,
            "missing": list(self.missing),
            "unexpected": list(self.unexpected),
            "mismatched": list(self.mismatched),
            "note": self.note,
        }


def _shape(tensor: torch.Tensor) -> Tuple[int, ...]:
    """Return a tensor's shape as a plain integer tuple."""
    return tuple(int(dim) for dim in tensor.shape)


def _graph_patch(graph: Any) -> Dict[str, torch.Tensor]:
    """Return the ``name.weight``/``name.bias`` tensors of a NIR graph."""
    patch: Dict[str, torch.Tensor] = {}
    for name, node in graph.nodes.items():
        for field in _WEIGHT_FIELDS:
            value = getattr(node, field, None)
            if value is None:
                continue
            patch[f"{name}.{field}"] = torch.as_tensor(value)
    return patch


def _mismatches(
    patch: Mapping[str, torch.Tensor],
    full: Mapping[str, torch.Tensor],
    required: Mapping[str, torch.Tensor],
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    """Return the ``(missing, unexpected, mismatched)`` key tuples.

    ``required`` is the set of keys the source must supply. A graph carries
    only weight/bias tensors, so neuron hyperparameter buffers are not
    required; a state dict must supply every stored key.
    """
    missing = tuple(sorted(key for key in required if key not in patch))
    unexpected = tuple(sorted(key for key in patch if key not in full))
    mismatched = tuple(sorted(
        key for key in patch
        if key in full and _shape(patch[key]) != _shape(full[key])
    ))
    return missing, unexpected, mismatched


def _note(
    missing: Sequence[str],
    unexpected: Sequence[str],
    mismatched: Sequence[str],
) -> str:
    """Return a human note naming every category of strict mismatch."""
    parts = []
    if missing:
        parts.append(f"missing {list(missing)}")
    if unexpected:
        parts.append(f"unexpected {list(unexpected)}")
    if mismatched:
        parts.append(f"shape mismatch {list(mismatched)}")
    return "; ".join(parts)


def _source(
    module: Any,
    state_dict: Optional[Mapping[str, Any]],
    graph: Optional[Any],
) -> Tuple[Dict[str, torch.Tensor], Mapping[str, torch.Tensor]]:
    """Return the ``(patch, required)`` pair for the supplied weight source."""
    if graph is not None:
        return _graph_patch(graph), dict(module.named_parameters())
    return dict(state_dict or {}), module.state_dict()


def _refused(
    missing: Tuple[str, ...],
    unexpected: Tuple[str, ...],
    mismatched: Tuple[str, ...],
) -> Optional[WeightLoadResult]:
    """Return a refusal naming the offending keys, or ``None`` when clean."""
    if not (missing or unexpected or mismatched):
        return None
    return WeightLoadResult(
        False, missing, unexpected, mismatched,
        _note(missing, unexpected, mismatched),
    )


def load_weights(
    module: Any,
    state_dict: Optional[Mapping[str, Any]] = None,
    graph: Optional[Any] = None,
) -> WeightLoadResult:
    """Map exactly one weight source onto ``module``, refusing a mismatch."""
    if (state_dict is None) == (graph is None):
        raise ValueError("provide exactly one weight source")
    patch, required = _source(module, state_dict, graph)
    missing, unexpected, mismatched = _mismatches(
        patch, module.state_dict(), required
    )
    refusal = _refused(missing, unexpected, mismatched)
    if refusal is not None:
        return refusal
    module.load_state_dict(patch, strict=False)
    return WeightLoadResult(
        True, (), (), (), f"loaded {len(patch)} tensors"
    )
