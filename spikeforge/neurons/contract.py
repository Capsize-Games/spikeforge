"""Canonical NIR parameter contract shared by every neuron handler.

Each handler converts its snnTorch constructor arguments into a flat, fully
typed dictionary of NIR-facing parameters. One timestep is one step
(``dt = 1``), which fixes the discrete membrane relation

    beta = exp(-dt / tau_mem)   <=>   tau_mem = -dt / ln(beta)

Every kind emits the same base keys:

``kind``         neuron kind string.
``dt``           timestep of the discrete relation (always ``1``).
``tau_mem``      membrane time constant derived from ``beta``.
``v_threshold``  firing threshold (from ``threshold``).
``v_reset``      reset potential: subtract -> ``v_threshold - 1``,
                 zero -> ``0``.
``v_leak``       leak potential (``0``).
``R``            membrane resistance ``1 / (1 - beta)``.

The resistance is the reconciliation that makes the discrete recurrences
coincide. ``nir.LIF``/``nir.LI`` are continuous-time nodes, so the reference
interpreter solves them with the exact zero-order-hold form

    v[t] = v_leak + (v[t-1] - v_leak) * decay + R * (1 - decay) * I[t]

with ``decay = exp(-dt / tau_mem) = beta``. snnTorch's discrete ``Leaky``
rule is ``mem[t] = beta * mem[t-1] + I[t]``, whose input enters with unit
gain, so the ZOH input coefficient ``R * (1 - beta)`` must equal ``1``:
hence ``R = 1 / (1 - beta)``. A naive ``R = 1`` would introduce a
``(1 - beta)`` mismatch and is not used.

Reset note (subtract): the installed NIR hard reset stores ``v_reset`` and
cannot retain the subtract residual, so ``v_reset`` above is only the reset
potential for the hard-reset ``LIF``/``CubaLIF`` nodes (and for snnTorch's
``zero`` mechanism, where ``v_reset = 0`` is exact). ``subtract`` reset is
reconciled exactly in :mod:`spikeforge.nir_bridge.neuron_nodes` with an
``LI`` integrator plus a ``Threshold``/``Delay``/``Scale`` feedback path.

``synaptic`` additionally emits ``tau_syn`` (derived from ``alpha``) and
``recurrent`` additionally emits the boolean ``feedback_delay`` flag. The
extras are additive, so the base keys stay identical across every kind.
"""

from math import exp, log
from typing import Any, Dict, Mapping, Tuple

import torch

#: A neuron's hidden state: an ordered tuple of tensors.
NeuronState = Tuple[torch.Tensor, ...]

DEFAULT_DT = 1.0
DEFAULT_BETA = 0.9
DEFAULT_ALPHA = 0.8
DEFAULT_THRESHOLD = 1.0
DEFAULT_RESET = "subtract"
V_LEAK = 0.0

BASE_KEYS = (
    "kind",
    "dt",
    "tau_mem",
    "v_threshold",
    "v_reset",
    "v_leak",
    "R",
)
SYNAPTIC_KEYS = ("tau_syn",)
RECURRENT_KEYS = ("feedback_delay",)
RESET_SUBTRACT = "subtract"
RESET_ZERO = "zero"
RESET_NONE = "none"


def get_float(
    params: Mapping[str, Any], key: str, default: float
) -> float:
    """Return ``params[key]`` coerced to ``float``, else ``default``."""
    return float(params.get(key, default))


def get_reset(params: Mapping[str, Any]) -> str:
    """Return the reset mechanism name, defaulting to subtract."""
    return str(params.get("reset", DEFAULT_RESET))


def tau_from_decay(decay: float, dt: float = DEFAULT_DT) -> float:
    """Return the time constant for ``decay = exp(-dt / tau)``."""
    return -dt / log(decay)


def resistance_from_decay(decay: float) -> float:
    """Return ``R = 1 / (1 - decay)`` so the ZOH input gain is exactly one.

    The reference interpreter solves the continuous NIR neuron equations
    with zero-order hold, giving an input coefficient ``R * (1 - decay)``.
    ``R = 1 / (1 - decay)`` cancels that factor so the discrete recurrence
    matches snnTorch's ``mem = decay * mem + input`` exactly. A ``decay``
    of one (no leak) has no finite resistance and is rejected.
    """
    gap = 1.0 - decay
    if gap <= 0.0:
        raise ValueError("decay must be below 1 for a finite resistance")
    return 1.0 / gap


def decay_from_tau(tau: float, dt: float = DEFAULT_DT) -> float:
    """Return ``exp(-dt / tau)`` for a time constant."""
    return exp(-dt / tau)


def reset_potential(reset: str, threshold: float) -> float:
    """Map a reset mechanism name to its NIR reset potential."""
    if reset == RESET_SUBTRACT:
        return threshold - 1.0
    if reset == RESET_ZERO:
        return 0.0
    raise ValueError(f"unknown reset mechanism: {reset!r}")


def base_params(kind: str, params: Mapping[str, Any]) -> Dict[str, Any]:
    """Build the shared NIR parameter dict for a neuron ``kind``."""
    dt = get_float(params, "dt", DEFAULT_DT)
    beta = get_float(params, "beta", DEFAULT_BETA)
    threshold = get_float(params, "threshold", DEFAULT_THRESHOLD)
    reset = get_reset(params)
    return {
        "kind": kind,
        "dt": dt,
        "tau_mem": tau_from_decay(beta, dt),
        "v_threshold": threshold,
        "v_reset": reset_potential(reset, threshold),
        "v_leak": V_LEAK,
        "R": resistance_from_decay(beta),
    }
