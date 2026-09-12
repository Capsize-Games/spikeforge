"""Inbound WebSocket message schema."""

from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field

from server.protocol_version import PROTOCOL_VERSION
from server.schemas.encode_config import EncodeConfig
from server.schemas.hub_query import HubQuery
from server.schemas.model_query import ModelQuery
from server.schemas.pipeline_graph import PipelineGraphConfig
from server.schemas.train_config import TrainConfig


class ClientMessage(BaseModel):
    """A message sent from the browser to the server."""

    protocol_version: str = PROTOCOL_VERSION
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
        "list_pipelines", "save_pipeline", "load_pipeline",
        "delete_pipeline", "run_pipeline", "stop_pipeline",
    ] = "configure"
    config: EncodeConfig = Field(default_factory=EncodeConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    query: ModelQuery = Field(default_factory=ModelQuery)
    hub: HubQuery = Field(default_factory=HubQuery)
    name: Optional[str] = None
    # Additive opt-in for the energy_report action: report the event-driven
    # sparse path (default) rather than the dense baseline.
    sparse: bool = True
    pipeline: PipelineGraphConfig = Field(default_factory=PipelineGraphConfig)
    # The raw {"frames": [...], "encoded": ...} request feeding a pipeline's
    # source node(s) -- separate from `pipeline` itself so a saved graph's
    # JSON never carries one run's throwaway input.
    pipeline_input: Dict[str, Any] = Field(default_factory=dict)
