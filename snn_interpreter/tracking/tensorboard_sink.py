"""A TensorBoard sink that records a manifest's scalars as an event file."""

import os
from typing import Any, Mapping, Optional

from snn_interpreter import config
from snn_interpreter.tracking import sink_probe, sink_records


class TensorBoardSink:
    """Write a tracking record's numeric fields to a TensorBoard directory."""

    def __init__(self, log_dir: Optional[str] = None) -> None:
        """Bind the sink to ``log_dir`` (default under the tracking root)."""
        base = log_dir or os.path.join(config.TRACKING_DIR, "tensorboard")
        self._log_dir = base

    @property
    def name(self) -> str:
        """Return the sink identifier."""
        return "tensorboard"

    @property
    def reason(self) -> str:
        """Return whether the extra is present."""
        if self.available():
            return "available"
        return "tracking extra 'tensorboard' is not installed"

    def available(self) -> bool:
        """Return True when TensorBoard can be imported."""
        return sink_probe.tensorboard_available()

    def log(self, record: Mapping[str, Any]) -> bool:
        """Write the record's scalars, returning False when unavailable."""
        writer_cls = sink_probe.tensorboard_writer()
        if writer_cls is None:
            return False
        writer = writer_cls(log_dir=self._log_dir)
        try:
            for key, value in sink_records.scalars(record).items():
                writer.add_scalar(key, value, 0)
        finally:
            writer.close()
        return True
