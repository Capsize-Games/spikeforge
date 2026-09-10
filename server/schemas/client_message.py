"""Inbound WebSocket message schema."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from server.schemas.encode_config import EncodeConfig
from server.schemas.train_config import TrainConfig


class ClientMessage(BaseModel):
    """A message sent from the browser to the server."""

    type: Literal[
        "configure", "run", "stop", "train", "stop_train", "predict",
        "select_sample", "infer", "save_model", "list_models",
        "load_model", "delete_model", "new_model", "stats",
        "cancel_download", "nir_export", "nir_validate",
        "trajectory", "metrics", "encoding_report",
        "surrogates", "surrogate_curve", "benchmark",
        "targets", "deployment_report",
    ] = "configure"
    config: EncodeConfig = Field(default_factory=EncodeConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    name: Optional[str] = None
