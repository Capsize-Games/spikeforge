"""Outbound WebSocket message schema."""

from typing import Any, Literal, Optional

from pydantic import BaseModel


class ServerMessage(BaseModel):
    """A message sent from the server to the browser."""

    type: Literal[
        "status", "image", "raster", "spike_frame",
        "curve", "error", "config_ack", "run_state",
        "train_metrics", "train_state", "prediction",
        "model_saved", "model_list", "model_loaded", "model_cleared",
        "inference", "system_stats", "download_state",
        "nir_graph", "nir_validation",
    ]
    payload: Any = None
    source: Optional[str] = None
