"""Delta (event-driven) spike coding (tutorial 2.4)."""

import torch
from snntorch import spikegen


class DeltaTrainer:
    """Encode a fake time-series tensor with spikegen.delta."""

    _data = None
    _spike_data = None
    _spike_data_off = None

    def __init__(self, threshold=4, off_spike=False, padding=False,
                 time_series=None):
        self._threshold = threshold
        self._off_spike = off_spike
        self._padding = padding
        self._data = torch.Tensor(
            [0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0]
            if time_series is None else time_series
        )
        self._apply_delta_coding()

    def _apply_delta_coding(self):
        """Encode the time series with and without off-spikes."""
        self._spike_data = spikegen.delta(
            self._data, threshold=self._threshold
        )
        self._spike_data_off = spikegen.delta(
            self._data, threshold=self._threshold, off_spike=True
        )

    @property
    def data(self):
        return self._data

    @property
    def spike_data(self):
        return self._spike_data

    @property
    def spike_data_off(self):
        return self._spike_data_off

    @property
    def threshold(self):
        return self._threshold

    @property
    def off_spike(self):
        return self._off_spike
