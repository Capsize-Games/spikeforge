"""Inbound WebSocket message schema."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from server.schemas.encode_config import EncodeConfig
from server.schemas.hub_query import HubQuery
from server.schemas.model_query import ModelQuery
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
        "targets", "deployment_report", "deploy_run", "energy_report",
        "model_search", "model_diff",
        "hub_list", "hub_search", "hub_download", "hub_cancel",
        "hub_inspect", "hub_import",
    ] = "configure"
    config: EncodeConfig = Field(default_factory=EncodeConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    query: ModelQuery = Field(default_factory=ModelQuery)
    hub: HubQuery = Field(default_factory=HubQuery)
    name: Optional[str] = None
    # Additive opt-in for the energy_report action: report the event-driven
    # sparse path (default) rather than the dense baseline.
    sparse: bool = True
