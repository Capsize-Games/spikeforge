"""WebSocket message schemas shared by server and client."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

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
    interval_ms: int = 100


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


class ClientMessage(BaseModel):
    """A message sent from the browser to the server."""

    type: Literal[
        "configure", "run", "stop", "train", "stop_train", "predict",
        "save_model", "list_models", "load_model", "delete_model",
    ] = "configure"
    config: EncodeConfig = Field(default_factory=EncodeConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    name: Optional[str] = None


class ServerMessage(BaseModel):
    """A message sent from the server to the browser."""

    type: Literal[
        "status", "image", "raster", "spike_frame",
        "curve", "error", "config_ack", "run_state",
        "train_metrics", "train_state", "prediction",
        "model_saved", "model_list", "model_loaded",
    ]
    payload: Any = None
