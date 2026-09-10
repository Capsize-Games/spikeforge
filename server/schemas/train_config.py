"""Training run configuration schema."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from server.schemas.encode_config import EncodeConfig


class TrainConfig(BaseModel):
    """User-adjustable parameters for a training run."""

    dataset: str = "mnist"
    hidden: int = 256
    beta: float = 0.9
    lr: float = 5e-3
    epochs: int = 3
    num_steps: int = 25
    subset: int = 10
    batch_size: int = 64
    checkpoint: Optional[str] = None  # load this model to continue
    device: Literal["auto", "cpu", "gpu"] = "auto"
    # Encoding settings for spike-input training; num_steps wins over the
    # legacy field above when a coding other than "raw" is active.
    encode: EncodeConfig = Field(default_factory=EncodeConfig)
