"""A trained trunk plus the contract needed to serve it."""

from dataclasses import dataclass
from typing import Dict, Tuple

from spikeforge.serving.encode_spec import EncodeSpec
from spikeforge.streaming.recipe_config import StreamTrainConfig
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule


@dataclass(frozen=True)
class StreamTrainingResult:
    """A trained trunk plus the contract needed to serve it."""

    spec: TopologySpec
    module: StageModule
    config: StreamTrainConfig
    encode_spec: EncodeSpec
    history: Tuple[Dict[str, float], ...]
