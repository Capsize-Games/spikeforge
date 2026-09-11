"""Export a built topology module's single forward step to ONNX.

The declared :class:`~spikeforge.topology.spec.TopologySpec` is rendered
once more as an ONNX graph of *one* time step; the temporal loop stays in the
simulator. The spec and the temporal contract are stamped into the model's
metadata so a re-import rebuilds the same topology exactly.
"""

from typing import Any, Dict, Optional

import torch

from spikeforge.onnx_bridge import api, metadata, step_module
from spikeforge.onnx_bridge.errors import OnnxExportError
from spikeforge.topology.registry import build_topology
from spikeforge.topology.spec import TopologySpec
from spikeforge.topology.stage_module import StageModule

#: Default batch used for the synthetic export input.
DEFAULT_BATCH = 2
#: Side used for a spatial entry stage when no explicit shape is supplied.
DEFAULT_SIDE = 28


def _dense_features(spec: TopologySpec) -> Optional[int]:
    """Return the feature width of the first dense stage, if any."""
    for stage in spec.stages:
        if stage.kind == "linear":
            return int(stage.params["in_features"])
    return None


def sample_input(
    spec: TopologySpec, batch: int = DEFAULT_BATCH
) -> torch.Tensor:
    """Return a deterministic input shaped for ``spec``'s entry stage."""
    entry = spec.stage(spec.input)
    if entry.kind == "conv2d":
        channels = int(entry.params.get("in_channels", 1))
        return torch.rand(batch, channels, DEFAULT_SIDE, DEFAULT_SIDE)
    if entry.kind == "conv1d":
        channels = int(entry.params.get("in_channels", 1))
        return torch.rand(batch, channels, DEFAULT_SIDE)
    features = _dense_features(spec)
    if features is None:
        raise OnnxExportError(f"no sample shape for entry kind {entry.kind!r}")
    return torch.rand(batch, features)


def _report(
    path: str,
    topology: str,
    model: Any,
    opset: int,
    sample: torch.Tensor,
) -> Dict[str, Any]:
    """Return the JSON-able export report for one written model."""
    return {
        "path": path,
        "topology": topology,
        "opset": opset,
        "ops": api.op_types(model),
        "nodes": len(model.graph.node),
        "input_shape": [int(dim) for dim in sample.shape],
        "temporal": metadata.TEMPORAL_CONTRACT,
    }


def export_built(
    spec: TopologySpec,
    module: StageModule,
    path: str,
    topology: str = "",
    batch: int = DEFAULT_BATCH,
    opset: int = api.DEFAULT_OPSET,
) -> Dict[str, Any]:
    """Export ``module``'s one-step forward to ``path`` with metadata."""
    sample = sample_input(spec, batch)
    adapter = step_module.StepModule(module, spec.output)
    api.export_onnx(adapter, (sample,), path, opset=opset)
    model = api.load_onnx(path)
    api.set_metadata(model, metadata.encode(spec, topology))
    api.save_onnx(model, path)
    return _report(path, topology, model, opset, sample)


def export_topology(
    topology: str,
    path: str,
    batch: int = DEFAULT_BATCH,
    opset: int = api.DEFAULT_OPSET,
) -> Dict[str, Any]:
    """Build ``topology`` and export its one-step forward to ``path``."""
    spec, module = build_topology(topology)
    return export_built(spec, module, path, topology, batch, opset)


def export_summary(path: str) -> Dict[str, Any]:
    """Return the op inventory and metadata of the model at ``path``."""
    model = api.load_onnx(path)
    return {
        "path": path,
        "ops": api.op_types(model),
        "nodes": len(model.graph.node),
        "metadata": api.metadata_values(model),
    }
