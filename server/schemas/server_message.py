"""Outbound WebSocket message schema."""

from typing import Any, Literal, Optional

from pydantic import BaseModel

from server.protocol_version import PROTOCOL_VERSION


class ServerMessage(BaseModel):
    """A message sent from the server to the browser."""

    protocol_version: str = PROTOCOL_VERSION
    type: Literal[
        "status", "image", "raster", "spike_frame",
        "curve", "error", "config_ack", "run_state",
        "train_metrics", "train_state", "prediction",
        "model_saved", "model_list", "model_loaded", "model_cleared",
        "inference", "system_stats", "download_state",
        "nir_graph", "nir_validation",
        "trajectory", "metrics", "encoding_report",
        "surrogate_list", "surrogate_curve", "benchmark",
        "target_list", "deployment_report", "backend_run",
        "energy_report", "animation_state",
        "hub_list", "hub_search", "hub_download_state",
        "hub_inspect", "hub_import",
    ]
    payload: Any = None
    source: Optional[str] = None
    # Undeclared on the wire today; made explicit so the schema can see it.
    kind: Optional[str] = None
