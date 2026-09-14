"""Latency (temporal) spike coding for MNIST (tutorial 2.3)."""

from typing import Any, Dict, Optional

import torch
from snntorch import spikegen

from spikeforge.training.trainer import SNNTrainer


def convert_to_time(
    data: torch.Tensor, tau: float = 5, threshold: float = 0.01
) -> torch.Tensor:
    """Map input intensity in [0,1] to an RC spike time (tutorial)."""
    return tau * torch.log(data / (data - threshold))


class LatencyTrainer(SNNTrainer):
    """Encode an MNIST batch via spikegen.latency across variants."""

    _tau: float = 5.0
    _threshold: float = 0.01
    _latency_steps: int = 100

    _latency_data: Optional[Dict[str, torch.Tensor]] = None
    _latency_targets: Optional[torch.Tensor] = None
    _latency_input: Optional[torch.Tensor] = None

    def __init__(
        self,
        tau: float = 5.0,
        threshold: float = 0.01,
        latency_steps: int = 100,
        **kwargs: Any,
    ) -> None:
        """Store latency options, then build the parent trainer."""
        self._tau = tau
        self._threshold = threshold
        self._latency_steps = latency_steps
        super().__init__(**kwargs)
        self._apply_latency_coding()

    def _kwargs(
        self,
        clip: bool = False,
        normalize: bool = False,
        linear: bool = False,
    ) -> Dict[str, Any]:
        """Build the spikegen.latency argument dict for a variant."""
        return {
            "num_steps": self._latency_steps,
            "tau": self._tau,
            "threshold": self._threshold,
            "clip": clip,
            "normalize": normalize,
            "linear": linear,
        }

    def _encode(
        self,
        data_it: torch.Tensor,
        clip: bool = False,
        normalize: bool = False,
        linear: bool = False,
    ) -> torch.Tensor:
        """Rate-code one batch with the requested latency flags."""
        return spikegen.latency(
            data_it, **self._kwargs(clip, normalize, linear)
        )

    def _apply_latency_coding(self) -> None:
        """Encode the first loader batch with all flag variants."""
        if self._train_loader is None:
            return
        data_it, targets_it = next(iter(self._train_loader))
        self._latency_data = {
            "base": self._encode(data_it),
            "linear": self._encode(data_it, linear=True),
            "normalized": self._encode(data_it, normalize=True, linear=True),
            "clip": self._encode(
                data_it, clip=True, normalize=True, linear=True
            ),
        }
        self._latency_targets = targets_it
        self._latency_input = data_it

    @property
    def latency_data(self) -> Optional[Dict[str, torch.Tensor]]:
        """Return the encoded spikes for each latency variant."""
        return self._latency_data

    @property
    def latency_targets(self) -> Optional[torch.Tensor]:
        """Return the labels for the encoded batch."""
        return self._latency_targets

    @property
    def latency_input(self) -> Optional[torch.Tensor]:
        """Return the raw input batch used for latency coding."""
        return self._latency_input

    @property
    def tau(self) -> float:
        """Return the membrane time constant."""
        return self._tau

    @property
    def threshold(self) -> float:
        """Return the latency-coding threshold."""
        return self._threshold

    @property
    def latency_steps(self) -> int:
        """Return the number of latency-coding time steps."""
        return self._latency_steps
