"""Resolve an optional ``surrogate`` build parameter to a ``spike_grad``.

Surrogate gradients only affect training, so they are deliberately kept out
of :func:`spikeforge.neurons.contract.base_params` and therefore out of
the NIR parameter contract. ``None`` (the default) reproduces the current
neurons exactly, because snnTorch substitutes its own default surrogate
whenever ``spike_grad`` is ``None``.
"""

from typing import Any, Callable, Mapping, Optional

import torch

from spikeforge.introspection.surrogate import resolve_surrogate

#: A callable in the ``spike_grad`` position: a tensor in, spikes out.
SpikeGrad = Callable[[torch.Tensor], torch.Tensor]


def spike_grad_from(params: Mapping[str, Any]) -> Optional[SpikeGrad]:
    """Return the selected surrogate's ``spike_grad``, or ``None``."""
    name = params.get("surrogate")
    if name is None:
        return None
    return resolve_surrogate(str(name))
