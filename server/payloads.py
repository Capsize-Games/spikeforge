"""Assemble outbound payloads from engines and encoder configs."""

from typing import Any, Dict, Optional

import torch

from server.schemas import EncodeConfig
from snn_interpreter.network import model_store
from snn_interpreter.nir_bridge import graph_summary, to_nir, validate
from snn_interpreter.topology.spec import TopologySpec


def compatibility(
    engine: Any, encode: Optional[EncodeConfig]
) -> Dict[str, Any]:
    """Compare a checkpoint's training config to the encoder config."""
    expected = engine.input_mode
    current = encode.coding if encode is not None else "raw"
    dataset = encode.dataset if encode is not None else engine.dataset
    steps = encode.num_steps if encode is not None else engine.num_steps
    return {
        "dataset_match": engine.dataset == dataset,
        "coding_match": expected == current,
        "num_steps_match": int(engine.num_steps) == int(steps),
        "expected_input_mode": expected,
        "current_coding": current,
    }


def model_loaded_payload(engine: Any, name: Optional[str],
                         encode: Optional[EncodeConfig],
                         accuracy: float) -> Dict[str, Any]:
    """Build the model_loaded payload including config coupling details."""
    return {
        "name": name,
        "dataset": engine.dataset,
        "accuracy": accuracy,
        "input_mode": engine.input_mode,
        "coding": engine.input_mode,
        "hidden": engine.hidden,
        "beta": engine.beta,
        "num_steps": engine.num_steps,
        "num_classes": engine.num_classes,
        "topology": engine.topology,
        "device": engine.device,
        "meta": _meta(name),
        "compatibility": compatibility(engine, encode),
    }


def nir_graph_payload(spec: TopologySpec, module: Any) -> Dict[str, Any]:
    """Return a JSON-able node/edge summary of a topology's NIR graph."""
    return graph_summary(to_nir(spec, module))


def nir_validation_payload(
    spec: TopologySpec,
    module: Any,
    spikes: torch.Tensor,
    graph: Optional[Any] = None,
) -> Dict[str, Any]:
    """Return a JSON-able drift report for a topology against its graph."""
    return validate(spec, module, _on_device(module, spikes), graph=graph)


def _on_device(module: Any, spikes: torch.Tensor) -> torch.Tensor:
    """Move ``spikes`` onto the module's device for the validator."""
    param = next(module.parameters(), None)
    return spikes if param is None else spikes.to(param.device)


def _meta(name: Optional[str]) -> Dict[str, Any]:
    """Read a checkpoint's stored meta, tolerating missing files."""
    try:
        return model_store.load(name).get("meta", {})
    except Exception:
        return {}
