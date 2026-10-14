"""Prune then compress a checkpoint's state dict for a bundle."""

from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from spikeforge.compression import (
    SCHEME_NONE,
    compress_state_dict,
    dequantize_state_dict,
    prune,
)

#: (stored, encoding, pruning, float weights) returned by ``_prepare_weights``.
WeightsPrep = Tuple[
    Dict[str, Any],
    Optional[Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, torch.Tensor],
]


def _apply_pruning(
    resolved: Dict[str, Any], prune_sparsity: Optional[float], strategy: str
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """Prune ``resolved`` when requested; return it and the report block."""
    if prune_sparsity is None:
        return resolved, None
    pruned = prune(resolved, prune_sparsity, strategy=strategy)
    return dict(pruned.tensors), pruned.report.to_dict()


def _uncompressed_weights(
    resolved: Dict[str, Any],
) -> Tuple[Dict[str, Any], None, Dict[str, torch.Tensor]]:
    """Return ``resolved`` unchanged, with its float tensors picked out."""
    floats = {
        name: value
        for name, value in resolved.items()
        if torch.is_tensor(value)
    }
    return resolved, None, floats


def _compressed_weights(
    resolved: Dict[str, Any], compress: str, compress_bits: int
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, torch.Tensor]]:
    """Compress ``resolved`` and return the stored/encoding/float triple."""
    compressed = compress_state_dict(resolved, compress, compress_bits)
    encoding = dict(compressed.encoding)
    stored = dict(compressed.tensors)
    return stored, encoding, dequantize_state_dict(stored, encoding)


def _prepare_weights(
    state_dict: Mapping[str, Any],
    compress: Optional[str],
    compress_bits: int,
    prune_sparsity: Optional[float],
    prune_strategy: str,
) -> WeightsPrep:
    """Prune then compress ``state_dict``; return stored/encoding/floats."""
    resolved, pruning = _apply_pruning(
        dict(state_dict), prune_sparsity, prune_strategy
    )
    if not compress or compress == SCHEME_NONE:
        stored, encoding, floats = _uncompressed_weights(resolved)
    else:
        stored, encoding, floats = _compressed_weights(
            resolved, compress, compress_bits
        )
    return stored, encoding, pruning, floats
