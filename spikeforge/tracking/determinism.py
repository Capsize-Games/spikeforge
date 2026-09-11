"""Opt-in deterministic mode and a bit-exactness fixture check.

Enabling determinism fixes every seed this project controls and turns on
PyTorch's deterministic-algorithm flags. That narrows, but does not close, the
bit-exactness gap: some CUDA/cuDNN kernels have no deterministic implementation
and hardware thread scheduling is outside the process's control.
:func:`enable_deterministic` therefore returns a report of what it did and did
not enforce instead of asserting universal bit-exactness, and
:func:`disable_deterministic` restores the flags so the fast default resumes.
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

#: cuBLAS reads this at initialisation to permit deterministic GEMMs.
_CUBLAS = "CUBLAS_WORKSPACE_CONFIG"
_CUBLAS_VALUE = ":4096:8"

_PRIOR: Optional[Dict[str, bool]] = None


@dataclass
class DeterminismReport:
    """What deterministic mode could and could not enforce for one run."""

    seed: Optional[int]
    applied: Dict[str, bool]
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Return the JSON-able report recorded in the manifest."""
        return {
            "enabled": True,
            "seed": self.seed,
            "exact_fixture": None,
            "applied": dict(self.applied),
            "notes": list(self.notes),
        }


def _remember_prior() -> None:
    """Capture the backend flags once so they can be restored exactly."""
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = {
            "use_deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        }


def _seed_value(seed: Optional[int]) -> int:
    """Return a non-negative 32-bit seed, defaulting to zero."""
    return 0 if seed is None else int(seed) % (2 ** 32)


def _report(seed: Optional[int]) -> DeterminismReport:
    """Summarise what was applied and what stays hardware-dependent."""
    applied = {
        "python": True,
        "numpy": True,
        "torch": True,
        "cublas_workspace": os.environ.get(_CUBLAS) == _CUBLAS_VALUE,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": not bool(torch.backends.cudnn.benchmark),
        "warn_only_ops": True,
    }
    notes = [
        "torch.use_deterministic_algorithms(warn_only=True) is set",
        "hardware thread scheduling is reported, not enforced",
    ]
    if not torch.cuda.is_available():
        notes.append("no CUDA device; the CPU path is deterministic here")
    return DeterminismReport(seed=seed, applied=applied, notes=notes)


def enable_deterministic(seed: Optional[int] = None) -> DeterminismReport:
    """Seed every RNG and enable PyTorch's deterministic algorithms."""
    _remember_prior()
    os.environ.setdefault(_CUBLAS, _CUBLAS_VALUE)
    value = _seed_value(seed)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return _report(seed)


def disable_deterministic() -> None:
    """Restore the torch backend flags captured before determinism was on."""
    global _PRIOR
    prior = _PRIOR or {
        "use_deterministic": False,
        "cudnn_deterministic": False,
        "cudnn_benchmark": False,
    }
    torch.use_deterministic_algorithms(bool(prior["use_deterministic"]))
    torch.backends.cudnn.deterministic = bool(prior["cudnn_deterministic"])
    torch.backends.cudnn.benchmark = bool(prior["cudnn_benchmark"])
    _PRIOR = None


def _as_tensor(value: Any) -> torch.Tensor:
    """Coerce a fixture's output to a tensor for comparison."""
    return value if isinstance(value, torch.Tensor) else torch.as_tensor(value)


def bit_exactness_check(
    run_a: Callable[[], Any], run_b: Callable[[], Any]
) -> Dict[str, Any]:
    """Run two fixtures and report whether their outputs match exactly."""
    first = _as_tensor(run_a())
    second = _as_tensor(run_b())
    if first.shape != second.shape:
        return {"exact": False, "max_abs_diff": None,
                "notes": ["output shapes differ"]}
    diff = float((first - second).abs().max().item()) if first.numel() else 0.0
    notes = [] if diff == 0.0 else ["outputs differ; see max_abs_diff"]
    return {"exact": diff == 0.0, "max_abs_diff": diff, "notes": notes}
