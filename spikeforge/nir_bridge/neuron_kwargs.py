"""Convert canonical NIR neuron parameters into ``nir`` kwargs.

The canonical contract uses ``tau_mem``/``R``; ``nir.LIF`` and ``nir.LI``
spell those ``tau``/``r``, so the rename lives here. Every value is a
scalar ``float32`` array because the NIR neuron nodes require all their
parameters to share a shape.

``nir.CubaLIF`` scales its input spike by ``w_in`` before integrating the
synaptic current. With zero-order hold the synaptic input gain is
``w_in * (1 - exp(-dt / tau_syn))``, so ``w_in = 1 / (1 - alpha)`` makes it
exactly one and matches snnTorch's ``syn = alpha * syn + input``.
"""

from math import exp
from typing import Any, Dict, Mapping

import numpy as np


def _scalar(value: Any) -> np.ndarray:
    """Return ``value`` as a scalar float32 array."""
    return np.array(float(value), dtype=np.float32)


def lif_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``nir.LIF`` keyword arguments for canonical ``params``."""
    return {
        "tau": _scalar(params["tau_mem"]),
        "r": _scalar(params["R"]),
        "v_leak": _scalar(params["v_leak"]),
        "v_threshold": _scalar(params["v_threshold"]),
        "v_reset": _scalar(params["v_reset"]),
    }


def li_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``nir.LI`` keyword arguments for canonical ``params``."""
    return {
        "tau": _scalar(params["tau_mem"]),
        "r": _scalar(params["R"]),
        "v_leak": _scalar(params["v_leak"]),
    }


def cuba_kwargs(params: Mapping[str, Any]) -> Dict[str, Any]:
    """Return ``nir.CubaLIF`` keyword arguments for canonical ``params``."""
    tau_syn = float(params["tau_syn"])
    dt = float(params.get("dt", 1.0))
    decay = exp(-dt / tau_syn)
    return {
        "tau_syn": _scalar(tau_syn),
        "tau_mem": _scalar(params["tau_mem"]),
        "r": _scalar(params["R"]),
        "v_leak": _scalar(params["v_leak"]),
        "v_threshold": _scalar(params["v_threshold"]),
        "v_reset": _scalar(params["v_reset"]),
        "w_in": _scalar(1.0 / (1.0 - decay)),
    }
