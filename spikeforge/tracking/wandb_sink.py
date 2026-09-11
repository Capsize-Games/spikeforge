"""A Weights & Biases sink that forwards a manifest to a W&B run."""

from typing import Any, Mapping

from spikeforge.tracking import sink_probe, sink_records


class WandBSink:
    """Log a tracking record to Weights & Biases when the extra is present."""

    def __init__(self, project: str = "spikeforge") -> None:
        """Store the project a W&B run is recorded under."""
        self._project = project

    @property
    def name(self) -> str:
        """Return the sink identifier."""
        return "wandb"

    @property
    def reason(self) -> str:
        """Return whether the extra is present."""
        if self.available():
            return "available"
        return "tracking extra 'wandb' is not installed"

    def available(self) -> bool:
        """Return True when ``wandb`` can be imported."""
        return sink_probe.wandb_available()

    def log(self, record: Mapping[str, Any]) -> bool:
        """Forward the record as a W&B run; False when unavailable."""
        module = sink_probe.wandb_module()
        if module is None:
            return False
        try:
            module.init(
                project=self._project,
                config=sink_records.summary(record),
                reinit=True,
            )
            module.log(sink_records.scalars(record))
            module.finish()
        except Exception:
            return False
        return True
