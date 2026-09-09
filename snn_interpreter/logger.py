"""Logging-aware subclass of :class:`SSNTrainer`."""

from snn_interpreter.trainer import SSNTrainer


class SNNTrainerLogger(SSNTrainer):
    """SSNTrainer variant that prints diagnostics during setup."""

    def _log(self, message):
        print(message)

    def _prepare_data(self):
        super()._prepare_data()
        self._log(f"MNIST training subset size: {self._subset_size}")

    def _prepare_vectorization(self):
        super()._prepare_vectorization()
        self._log(f"Converted vector: {self._rate_coded_vector}")
        pct = self._rate_coded_vector.sum() * 100 / len(
            self._rate_coded_vector
        )
        self._log(f"The output is spiking {pct:.2f}% of the time")

    def _apply_rate_coding(self):
        super()._apply_rate_coding()
        self._log(
            f"The shape of the rate-coded spike_data is "
            f"{self._spike_data.size()}"
        )
        self._log(f"The corresponding target is: {self._spike_targets[0]}")
