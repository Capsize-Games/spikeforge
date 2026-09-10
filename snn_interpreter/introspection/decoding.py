"""Approximate image inverses for each spike coding.

Each function is a pure inverse of the matching :class:`SpikeEncoder` coding
and returns a flat ``[F]`` float tensor in the encoder's input intensity
scale. The inverses are approximations, documented per coding:

``rate``
    ``spikegen.rate`` samples each spike from a Bernoulli distribution whose
    mean is the clamped input intensity, so the time-average of the spike
    train is an unbiased but noisy estimate; it converges as ``num_steps``
    grows.

``latency``
    Time-to-first-spike is quantised to integer steps and optionally
    rescaled to ``num_steps``, so inverting the time mapping recovers the
    intensity only up to that quantisation. Intensities at or below
    ``threshold`` share one clamped spike time and decode to the threshold
    ceiling rather than their true value.

``delta``
    ``spikegen.delta`` emits one spike per one-step increase at or above the
    change threshold, regardless of how large the increase is. Integrating
    the on/off stream credits exactly one threshold of change per spike, so
    the integral is a lower bound that is exact only when a step rises by
    exactly the threshold; slower pixels emit nothing and decode to zero.
"""

import math
from typing import Tuple

import torch

_LOG_EPSILON = 1e-7


def first_spike_indices(
    spikes: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return ``(index, fired)`` per flattened pixel of a ``[T, ..., F]``."""
    flat = spikes.detach().reshape(spikes.size(0), -1)
    fired = flat.sum(dim=0) > 0
    index = flat.argmax(dim=0).float()
    return index, fired


def decode_rate(spikes: torch.Tensor) -> torch.Tensor:
    """Return the mean spike count over time as the intensity estimate."""
    return spikes.detach().float().mean(dim=0).reshape(-1)


def decode_delta(spikes: torch.Tensor, threshold: float) -> torch.Tensor:
    """Integrate the on/off stream and credit ``threshold`` per spike."""
    integrated = spikes.detach().float().cumsum(dim=0)[-1].reshape(-1)
    return integrated * float(threshold)


def decode_latency(
    spikes: torch.Tensor,
    num_steps: int,
    tau: float,
    threshold: float,
    normalize: bool,
    linear: bool,
) -> torch.Tensor:
    """Invert the latency time-to-first-spike mapping per pixel."""
    index, fired = first_spike_indices(spikes)
    if linear:
        intensity = _linear_intensity(index, num_steps, tau, normalize)
    else:
        intensity = _log_intensity(
            index, num_steps, tau, threshold, normalize
        )
    return torch.where(fired, intensity, torch.zeros_like(intensity))


def _linear_intensity(
    index: torch.Tensor, num_steps: int, tau: float, normalize: bool
) -> torch.Tensor:
    """Invert the linear code ``t = span * (1 - intensity)``."""
    span = float(num_steps - 1) if normalize else float(tau)
    if span <= 0.0:
        return torch.zeros_like(index)
    return torch.clamp(1.0 - index / span, 0.0, 1.0)


def _log_intensity(
    index: torch.Tensor,
    num_steps: int,
    tau: float,
    threshold: float,
    normalize: bool,
) -> torch.Tensor:
    """Invert the log code ``t = tau * log(d / (d - threshold))``."""
    scale = _log_normaliser(num_steps, tau, threshold) if normalize else 1.0
    decay = torch.exp(-(index * scale) / float(tau))
    return torch.clamp(
        float(threshold) / (1.0 - decay + _LOG_EPSILON), 0.0, 1.0
    )


def _log_normaliser(num_steps: int, tau: float, threshold: float) -> float:
    """Return the batch normaliser spikegen applies when normalising.

    ``spikegen`` rescales the raw spike times so the maximum equals
    ``num_steps - 1``; the maximum raw time is produced by the clamped
    minimum intensity ``threshold + epsilon``.
    """
    span = float(num_steps - 1)
    if span <= 0.0:
        return 0.0
    peak = float(tau) * math.log(
        (float(threshold) + _LOG_EPSILON) / _LOG_EPSILON
    )
    return peak / span
