"""The geometry, encode contract, and optimiser for one training run."""

from dataclasses import dataclass
from typing import Any, Dict

from spikeforge.serving.encode_spec import EncodeSpec

#: Topology the recipe trains; the NIR-mappable sequence preset.
TOPOLOGY: str = "sequence_mlp"


@dataclass(frozen=True)
class StreamTrainConfig:
    """The geometry, encode contract, and optimiser for one training run."""

    seq_length: int = 16
    features: int = 4
    hidden: int = 16
    num_classes: int = 3
    num_steps: int = 6
    coding: str = "delta"
    delta_threshold: float = 4.0
    delta_along_window: bool = False
    lr: float = 5e-2
    epochs: int = 20
    batch_size: int = 8
    seed: int = 0

    def validate(self) -> None:
        """Raise :class:`ValueError` when the config is unusable."""
        if min(self.seq_length, self.features, self.hidden) < 1:
            raise ValueError("seq_length, features and hidden must be >= 1")
        if self.num_classes < 2 or self.num_steps < 1:
            raise ValueError(
                "num_classes >= 2 and num_steps >= 1 are required"
            )
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be >= 1")
        self.encode_spec().validate()

    def encode_spec(self) -> EncodeSpec:
        """Return the frozen encode contract both train and serve share."""
        return EncodeSpec(
            coding=self.coding,
            num_steps=self.num_steps,
            delta_threshold=self.delta_threshold,
            random_seed=self.seed,
        )

    def topology_params(self) -> Dict[str, Any]:
        """Return the ``sequence_mlp`` preset parameters for this config."""
        return {
            "seq_length": int(self.seq_length),
            "features": int(self.features),
            "hidden": int(self.hidden),
            "num_classes": int(self.num_classes),
        }
