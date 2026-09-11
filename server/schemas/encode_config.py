"""Encoding experiment configuration schema."""

from typing import Literal, Optional, Tuple

from pydantic import BaseModel

CodingType = Literal["rate", "latency", "delta", "random"]


class EncodeConfig(BaseModel):
    """User-adjustable parameters for an encoding experiment."""

    coding: CodingType = "rate"
    dataset: str = "mnist"
    subset: int = 10
    batch_size: int = 128
    num_steps: int = 100
    sample_index: int = 0
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
    interval_ms: int = 100
    #: Optional (H, W) sensor geometry; None keeps the default 28x28.
    input_size: Optional[Tuple[int, int]] = None
    #: Opt-in per-step hidden-layer animation; default off keeps the stream.
    animate_hidden: bool = False
