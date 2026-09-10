"""Latency (temporal) spike coding for MNIST (tutorial 2.3)."""

import torch
from snntorch import spikegen

from snn_interpreter.trainer import SSNTrainer


def convert_to_time(data, tau=5, threshold=0.01):
    """Map input intensity in [0,1] to an RC spike time (tutorial)."""
    return tau * torch.log(data / (data - threshold))


class LatencyTrainer(SSNTrainer):
    """Encode an MNIST batch via spikegen.latency across variants."""

    _tau = 5.0
    _threshold = 0.01
    _latency_steps = 100

    _latency_data = None
    _latency_targets = None
    _latency_input = None

    def __init__(self, tau=5.0, threshold=0.01, latency_steps=100,
                 **kwargs):
        self._tau = tau
        self._threshold = threshold
        self._latency_steps = latency_steps
        super().__init__(**kwargs)
        self._apply_latency_coding()

    def _kwargs(self, clip=False, normalize=False, linear=False):
        """Build the spikegen.latency argument dict for a variant."""
        return dict(
            num_steps=self._latency_steps,
            tau=self._tau,
            threshold=self._threshold,
            clip=clip,
            normalize=normalize,
            linear=linear,
        )

    def _encode(self, data_it, clip=False, normalize=False, linear=False):
        """Rate-code one batch with the requested latency flags."""
        return spikegen.latency(
            data_it, **self._kwargs(clip, normalize, linear)
        )

    def _apply_latency_coding(self):
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
    def latency_data(self):
        return self._latency_data

    @property
    def latency_targets(self):
        return self._latency_targets

    @property
    def latency_input(self):
        return self._latency_input

    @property
    def tau(self):
        return self._tau

    @property
    def threshold(self):
        return self._threshold

    @property
    def latency_steps(self):
        return self._latency_steps
