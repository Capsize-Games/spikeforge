"""Encoding experiment configuration schema."""

from typing import Annotated, Literal, Optional, Tuple

from pydantic import BaseModel, Field

CodingType = Literal["rate", "latency", "delta", "random"]

#: Bounds for one ``input_size`` (H, W) sensor dimension.
SensorDim = Annotated[int, Field(ge=1, le=1024)]


class EncodeConfig(BaseModel):
    """User-adjustable parameters for an encoding experiment."""

    coding: CodingType = "rate"
    dataset: str = "mnist"
    subset: int = Field(default=10, ge=1, le=1000000)
    batch_size: int = Field(default=128, ge=1, le=8192)
    num_steps: int = Field(default=100, ge=1, le=100000)
    sample_index: int = Field(default=0, ge=0)
    gain: float = 0.25
    vector_value: float = 0.5
    tau: float = 5.0
    threshold: float = 0.01
    clip: bool = False
    normalize: bool = True
    linear: bool = True
    off_spike: bool = False
    delta_threshold: float = 4.0
    random_scale: float = 0.5
    random_seed: Optional[int] = None
    interval_ms: int = Field(default=100, ge=0, le=3600000)
    #: Optional (H, W) sensor geometry; None keeps the default 28x28.
    input_size: Optional[Tuple[SensorDim, SensorDim]] = None
    #: Opt-in per-step hidden-layer animation; default off keeps the stream.
    animate_hidden: bool = False
