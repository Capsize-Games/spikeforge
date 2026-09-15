"""The symmetric fixed-point grid every simulated quantizer snaps onto.

A grid is defined by a signed code width (``bits``) and a magnitude bound: it
has ``2 ** (bits - 1) - 1`` levels on either side of zero, spaced
``bound / levels`` apart. Snapping rounds each value to the nearest level and
clips anything beyond the bound to the outermost level. That models the
resolution loss and the saturation of a fixed-point register; it does not
model integer accumulation, accumulator overflow, or any vendor kernel.
"""

import torch

#: Peak below which a tensor is treated as all-zero.
EPS = 1e-12


def levels_for(bits: int) -> float:
    """Return the largest code magnitude of a signed ``bits``-wide grid."""
    if bits <= 0:
        return 0.0
    return float(2 ** (bits - 1) - 1)


def snap(tensor: torch.Tensor, bound: float, levels: float) -> torch.Tensor:
    """Return ``tensor`` rounded onto ``levels`` steps up to ``bound``.

    A grid with no levels leaves the tensor untouched, and a bound at or
    below :data:`EPS` collapses it to zeros, because a grid whose span is
    zero cannot represent anything else. Nothing is mutated in place.
    """
    if levels <= 0.0:
        return tensor
    if bound <= EPS:
        return torch.zeros_like(tensor)
    scale = bound / levels
    codes = torch.round(tensor / scale).clamp(-levels, levels)
    return codes * scale
