"""Delta (event-driven) spike coding (tutorial 2.4)."""

from typing import Optional, Sequence

import torch
from snntorch import spikegen


class DeltaTrainer:
    """Encode a fake time-series tensor with spikegen.delta."""

    _data: Optional[torch.Tensor] = None
    _spike_data: Optional[torch.Tensor] = None
    _spike_data_off: Optional[torch.Tensor] = None

    def __init__(
        self,
        threshold: int = 4,
        off_spike: bool = False,
        padding: bool = False,
        time_series: Optional[Sequence[float]] = None,
    ) -> None:
        """Store the delta options and encode the demo time series."""
        self._threshold = threshold
        self._off_spike = off_spike
        self._padding = padding
        self._data = torch.Tensor(
            [0, 1, 0, 2, 8, -20, 20, -5, 0, 1, 0]
            if time_series is None
            else time_series
        )
        self._apply_delta_coding()

    def _apply_delta_coding(self) -> None:
        """Encode the time series with and without off-spikes."""
        self._spike_data = spikegen.delta(
            self._data, threshold=self._threshold
        )
        self._spike_data_off = spikegen.delta(
            self._data, threshold=self._threshold, off_spike=True
        )

    @property
    def data(self) -> Optional[torch.Tensor]:
        """Return the source time series."""
        return self._data

    @property
    def spike_data(self) -> Optional[torch.Tensor]:
        """Return the on-spike delta encoding."""
        return self._spike_data

    @property
    def spike_data_off(self) -> Optional[torch.Tensor]:
        """Return the off-spike delta encoding."""
        return self._spike_data_off

    @property
    def threshold(self) -> int:
        """Return the delta threshold."""
        return self._threshold

    @property
    def off_spike(self) -> bool:
        """Return the configured off-spike flag."""
        return self._off_spike
