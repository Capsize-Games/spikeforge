"""Pure error metrics between two per-step trajectories.

Every function returns plain JSON-serialisable numbers, never tensors, so a
validation report can be streamed unchanged. ``actual`` is the independent
NIR interpretation and ``reference`` the snnTorch trajectory; the relative
error is normalised by the reference magnitude.
"""

from typing import Any, Dict, Tuple

import numpy as np
import torch

#: A value above this level counts as a spike for the agreement fraction.
SPIKE_LEVEL = 0.5


def _array(values: Any) -> np.ndarray:
    """Return ``values`` as a float64 numpy array."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def _aligned(actual: Any, reference: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``actual`` and ``reference`` as shape-matching arrays."""
    left = _array(actual)
    right = _array(reference)
    if left.shape != right.shape:
        message = f"shape mismatch: {left.shape} vs {right.shape}"
        raise ValueError(message)
    return left, right


def max_abs_error(actual: Any, reference: Any) -> float:
    """Return the largest absolute elementwise difference."""
    left, right = _aligned(actual, reference)
    if left.size == 0:
        return 0.0
    return float(np.max(np.abs(left - right)))


def mean_abs_error(actual: Any, reference: Any) -> float:
    """Return the mean absolute elementwise difference."""
    left, right = _aligned(actual, reference)
    if left.size == 0:
        return 0.0
    return float(np.mean(np.abs(left - right)))


def relative_error(actual: Any, reference: Any) -> float:
    """Return the max absolute error divided by the reference peak."""
    left, right = _aligned(actual, reference)
    peak = float(np.max(np.abs(right))) if right.size else 0.0
    if peak <= 0.0:
        return max_abs_error(left, right)
    return max_abs_error(left, right) / peak


def agreement_fraction(actual: Any, reference: Any) -> float:
    """Return the fraction of positions that agree on spike presence."""
    left, right = _aligned(actual, reference)
    if left.size == 0:
        return 1.0
    matches = (left > SPIKE_LEVEL) == (right > SPIKE_LEVEL)
    return float(np.mean(matches))


def compare(actual: Any, reference: Any) -> Dict[str, float]:
    """Return every metric for ``actual`` against ``reference``."""
    return {
        "max_abs": max_abs_error(actual, reference),
        "mean_abs": mean_abs_error(actual, reference),
        "relative": relative_error(actual, reference),
        "agreement": agreement_fraction(actual, reference),
    }
