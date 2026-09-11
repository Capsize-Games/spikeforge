"""Zero-order-hold neuron and pointwise node operations."""

from typing import Any, Optional, Tuple

import torch

from spikeforge.nir_bridge.ops_common import as_tensor

#: One NIR time unit per interpreter step (the canonical ``dt = 1``).
DT = 1.0

Result = Tuple[torch.Tensor, Any, Optional[torch.Tensor]]


def zoh_state(
    v_prev: torch.Tensor,
    current: torch.Tensor,
    tau: torch.Tensor,
    r: torch.Tensor,
    v_leak: torch.Tensor,
) -> torch.Tensor:
    """Return one zero-order-hold step of ``tau*v' = (v_leak - v) + R*I``."""
    decay = torch.exp(-DT / tau)
    return v_leak + (v_prev - v_leak) * decay + r * (1.0 - decay) * current


def apply_identity(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return ``x`` unchanged (the ``Output`` node is a pass-through)."""
    return x, None, None


def apply_threshold(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return the strict Heaviside ``1[x > threshold]`` as a spike train."""
    spike = (x > as_tensor(node.threshold, x)).to(x.dtype)
    return spike, None, None


def apply_scale(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Return the Hadamard product ``x * scale``."""
    return x * as_tensor(node.scale, x), None, None


def apply_li(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Integrate ``x`` into the membrane; the output is the membrane."""
    v_prev = state if state is not None else torch.zeros_like(x)
    membrane = zoh_state(
        v_prev,
        x,
        as_tensor(node.tau, x),
        as_tensor(node.r, x),
        as_tensor(node.v_leak, x),
    )
    return membrane, membrane, membrane


def fire(
    membrane: torch.Tensor,
    threshold: torch.Tensor,
    reset: torch.Tensor,
    reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the hard-reset ``(spike, stored)`` pair for a membrane."""
    spike = (membrane > threshold).to(reference.dtype)
    stored = torch.where(spike > 0, reset, membrane)
    return spike, stored


def apply_lif(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Integrate with a hard reset; the output is the spike train."""
    v_prev = state if state is not None else torch.zeros_like(x)
    membrane = zoh_state(
        v_prev,
        x,
        as_tensor(node.tau, x),
        as_tensor(node.r, x),
        as_tensor(node.v_leak, x),
    )
    spike, stored = fire(
        membrane,
        as_tensor(node.v_threshold, x),
        as_tensor(node.v_reset, x),
        x,
    )
    return spike, stored, membrane


def apply_if(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Integrate without leak (``v += R*I``), then apply the hard reset.

    ``nir.IF`` has no leak term, unlike ``nir.LIF``, so it needs its own
    update rather than the zero-order-hold form used for leaky neurons.
    """
    v_prev = state if state is not None else torch.zeros_like(x)
    membrane = v_prev + as_tensor(node.r, x) * x
    spike, stored = fire(
        membrane,
        as_tensor(node.v_threshold, x),
        as_tensor(node.v_reset, x),
        x,
    )
    return spike, stored, membrane


def _cuba_prev(
    state: Any, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the previous ``(syn, mem)`` tensors for a CubaLIF step."""
    if state is None:
        return torch.zeros_like(x), torch.zeros_like(x)
    return state


def _cuba_membrane(
    node: Any, syn: torch.Tensor, mem_prev: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Integrate the CubaLIF membrane from the synaptic current."""
    return zoh_state(
        mem_prev,
        syn,
        as_tensor(node.tau_mem, x),
        as_tensor(node.r, x),
        as_tensor(node.v_leak, x),
    )


def apply_cuba(node: Any, x: torch.Tensor, state: Any) -> Result:
    """Integrate the synaptic current, then the membrane, with hard reset."""
    syn_prev, mem_prev = _cuba_prev(state, x)
    zero = torch.zeros((), dtype=x.dtype, device=x.device)
    syn = zoh_state(
        syn_prev, x, as_tensor(node.tau_syn, x), as_tensor(node.w_in, x), zero
    )
    membrane = _cuba_membrane(node, syn, mem_prev, x)
    spike, stored = fire(
        membrane,
        as_tensor(node.v_threshold, x),
        as_tensor(node.v_reset, x),
        x,
    )
    return spike, (syn, stored), membrane
