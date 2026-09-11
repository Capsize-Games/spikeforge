"""Encode and decode the bridge metadata attached to an exported model.

The ONNX graph is only one rendering of the topology, so an export stamps the
declarative spec and the temporal contract into the model's metadata
properties. A re-import can then rebuild the *same* topology exactly instead of
guessing stages back from ops. Third-party models without this metadata fall
back to the op mapping in :mod:`spikeforge.onnx_bridge.import_onnx`.
"""

import json
from typing import Dict, Mapping, Optional

from spikeforge.topology.spec import TopologySpec

#: Metadata key naming the exported topology.
TOPOLOGY_KEY = "snn_topology"
#: Metadata key carrying the JSON spec.
SPEC_KEY = "snn_spec"
#: Metadata key naming the temporal contract.
TEMPORAL_KEY = "snn_temporal"
#: The contract every export encodes: ONNX holds one step, not the loop.
TEMPORAL_CONTRACT = "single-step; the time loop stays in the simulator"


def encode(spec: TopologySpec, topology: str) -> Dict[str, str]:
    """Return the metadata properties describing ``spec``."""
    return {
        TOPOLOGY_KEY: topology,
        SPEC_KEY: json.dumps(spec.to_dict()),
        TEMPORAL_KEY: TEMPORAL_CONTRACT,
    }


def topology_of(values: Mapping[str, str]) -> Optional[str]:
    """Return the recorded topology name, or ``None`` when absent."""
    return values.get(TOPOLOGY_KEY)


def decode(values: Mapping[str, str]) -> Optional[TopologySpec]:
    """Rebuild the spec recorded in ``values``, or ``None`` when absent."""
    payload = values.get(SPEC_KEY)
    if not payload:
        return None
    try:
        return TopologySpec.from_dict(json.loads(payload))
    except (KeyError, TypeError, ValueError):
        return None
