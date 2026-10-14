"""Build the in-memory NIR graph envelope for a bundle."""

import json
import os
import tempfile
from typing import Any, Mapping, Tuple

import torch

from spikeforge.serving import bundle_manifest as bm
from spikeforge.serving.errors import BundleFormatError
from spikeforge.topology.builder import build_module
from spikeforge.topology.spec import TopologySpec


def _load_module(
    spec_dict: Mapping[str, Any], state_dict: Mapping[str, torch.Tensor]
) -> Tuple[TopologySpec, Any]:
    """Build the topology module and load its trained weights."""
    spec = TopologySpec.from_dict(spec_dict)
    module = build_module(spec)
    module.load_state_dict(dict(state_dict), strict=True)
    return spec, module


def _nir_envelope(
    spec_dict: Mapping[str, Any],
    state_dict: Mapping[str, torch.Tensor],
    source: str,
) -> Mapping[str, Any]:
    """Return the in-memory NIR graph envelope for the bundle."""
    from spikeforge.nir_bridge import api as nir_api
    from spikeforge.nir_bridge import serialization
    from spikeforge.nir_bridge.exporter import to_nir

    if not nir_api.available():
        raise BundleFormatError(
            source, "the 'nir' extra is required to include a NIR graph"
        )
    spec, module = _load_module(spec_dict, state_dict)
    with tempfile.TemporaryDirectory() as folder:
        graph_path = os.path.join(folder, bm.GRAPH_NAME)
        serialization.save_graph(to_nir(spec, module), graph_path)
        with open(graph_path, encoding="utf-8") as handle:
            return json.load(handle)
