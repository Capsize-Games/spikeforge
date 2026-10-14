"""Build NIR nodes for stages that decompose into one ``Affine`` node.

An ``embedding`` stage is a row gather, which is exactly a one-hot matrix
product: token ids expand to one-hot at ingest (see
:mod:`spikeforge.nir_bridge.one_hot`) and the lookup table becomes the
transposed weight of an ``Affine`` node with a zero bias. A 2D eval-mode
``batch_norm`` stage is a per-channel affine map once its frozen running
statistics fold into a diagonal weight and a shifted bias.

Both decompositions reuse the C-supported ``Affine`` primitive, so no new
NIR node kind is needed. Weighted exports copy their tensors from the
built submodule; structural exports (no module) carry coherent
placeholders -- a zero table for embedding, the fresh-module fold for
batch norm -- matching the linear-stage precedent.
"""

from typing import Any, Mapping, Optional, Tuple

import numpy as np

from spikeforge.nir_bridge.errors import UnsupportedStageError
from spikeforge.nir_bridge.mapped_node import MappedNode
from spikeforge.nir_bridge.require import require_node

#: Reason a ``dim=2`` batch norm stage stays simulation-only.
_BN2D_REASON = (
    "BatchNorm2d has no exact Affine fold: NIR Affine acts on the last "
    "dim and there is no reshape primitive"
)


def _np(tensor: Any) -> np.ndarray:
    """Detach a torch tensor into a numpy array."""
    return tensor.detach().cpu().numpy()


def embedding_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``Affine`` node rendering an embedding stage.

    The ``[num_embeddings, embedding_dim]`` table becomes the transposed
    weight with a zero bias, so a one-hot frame selects rows exactly.
    """
    vocab = int(params["num_embeddings"])
    width = int(params["embedding_dim"])
    if submodule is None:
        weight = np.zeros((width, vocab), np.float32)
    else:
        weight = _np(submodule.weight).T
    bias = np.zeros(width, dtype=weight.dtype)
    cls = require_node("Affine", "embedding")
    return MappedNode(name, cls(weight, bias))


def _ensure_foldable(submodule: Any) -> None:
    """Raise unless ``submodule`` is a foldable eval BatchNorm1d."""
    kind = type(submodule).__name__
    if kind != "BatchNorm1d":
        raise UnsupportedStageError(
            "batch_norm", f"only BatchNorm1d folds into Affine, not {kind}"
        )
    if bool(submodule.training):
        raise UnsupportedStageError(
            "batch_norm",
            "training-mode batch norm uses batch statistics, which are "
            "runtime data and cannot fold into static weights",
        )
    if not bool(submodule.track_running_stats):
        raise UnsupportedStageError(
            "batch_norm",
            "without running statistics the module uses batch "
            "statistics even in eval mode, which cannot fold",
        )


def _fold(
    mean: np.ndarray,
    var: np.ndarray,
    gamma: np.ndarray,
    beta: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the ``(weight, bias)`` Affine fold of norm buffers."""
    scale = gamma / np.sqrt(var + eps)
    return np.diag(scale), beta - mean * scale


def _module_fold(submodule: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return the fold of an eval BatchNorm1d's frozen buffers."""
    if submodule.weight is None:
        features = int(submodule.num_features)
        gamma = np.ones(features, np.float32)
        beta = np.zeros(features, np.float32)
    else:
        gamma = _np(submodule.weight)
        beta = _np(submodule.bias)
    return _fold(
        _np(submodule.running_mean),
        _np(submodule.running_var),
        gamma,
        beta,
        float(submodule.eps),
    )


def _structural_fold(
    params: Mapping[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the fresh-module fold for a structural export."""
    features = int(params["num_features"])
    eps = float(params.get("eps", 1e-5))
    ones = np.ones(features, np.float32)
    zeros = np.zeros(features, np.float32)
    return _fold(zeros, ones, ones, zeros, eps)


def batch_norm_node(
    name: str, params: Mapping[str, Any], submodule: Optional[Any]
) -> MappedNode:
    """Return the ``Affine`` node folding a 2D eval batch norm stage.

    BatchNorm2d and batch-statistic modules raise; the fold is exact
    for ``[B, C]`` inference inputs only.
    """
    if int(params.get("dim", 1)) == 2:
        raise UnsupportedStageError("batch_norm", _BN2D_REASON)
    if submodule is None:
        weight, bias = _structural_fold(params)
    else:
        _ensure_foldable(submodule)
        weight, bias = _module_fold(submodule)
    cls = require_node("Affine", "batch_norm")
    return MappedNode(name, cls(weight, bias))
