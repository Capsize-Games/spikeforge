"""Import an ONNX graph as a topology spec, or reject it by op name.

A model exported by this bridge carries its declarative spec in metadata, so
it re-imports exactly. A third-party model has no such metadata: its ops are
mapped to stage kinds only where a faithful SNN equivalent exists, and any
other op raises :class:`~spikeforge.onnx_bridge.errors.
UnsupportedOnnxOpError` naming it — a partial import is never returned.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

from spikeforge.onnx_bridge import api, metadata
from spikeforge.onnx_bridge.errors import UnsupportedOnnxOpError
from spikeforge.topology.spec import TopologySpec, chain
from spikeforge.topology.stage import Stage

#: An op -> stage builder.
OpBuilder = Callable[[Dict[str, Any], int], Stage]
#: ONNX ops the importer can represent as a stage.
PASSTHROUGH = frozenset({"Identity"})


def _pair(value: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """Normalise a scalar or length-2 sequence into an int pair."""
    if value is None:
        return default
    if isinstance(value, (int, float)):
        size = int(value)
        return (size, size)
    items = [int(item) for item in value]
    if len(items) == 1:
        return (items[0], items[0])
    return (items[0], items[1])


def _size(pair: Tuple[int, int]) -> Any:
    """Return a scalar for a square pair, else a two-element list."""
    return pair[0] if pair[0] == pair[1] else [pair[0], pair[1]]


def _padding(attrs: Dict[str, Any]) -> Any:
    """Return ONNX pads as an int when symmetric, else the raw tuple."""
    pads = [int(item) for item in attrs.get("pads", [])]
    if not pads:
        return 0
    return pads[0] if len(set(pads)) == 1 else tuple(pads)


def _name(op: Dict[str, Any], index: int) -> str:
    """Return a stable stage name for an ONNX node."""
    return str(op["name"]) or f"{str(op['op_type']).lower()}_{index}"


def _linear(op: Dict[str, Any], index: int) -> Stage:
    """Map a ``Gemm``/``MatMul`` node to a ``linear`` stage."""
    shape = op.get("weight_shape")
    if not shape or len(shape) != 2:
        raise UnsupportedOnnxOpError(
            op["op_type"], "no 2-D weight initializer to size the layer"
        )
    transposed = bool(int(op["attrs"].get("transB", 0) or 0))
    if transposed:
        out_features, in_features = shape[0], shape[1]
    else:
        in_features, out_features = shape[0], shape[1]
    params = {
        "in_features": int(in_features),
        "out_features": int(out_features),
    }
    return Stage(_name(op, index), "linear", params)


def _conv(op: Dict[str, Any], index: int) -> Stage:
    """Map a ``Conv`` node to a ``conv2d`` stage."""
    shape = op.get("weight_shape")
    if not shape or len(shape) != 4:
        raise UnsupportedOnnxOpError(
            op["op_type"], "no 4-D weight initializer to size the layer"
        )
    attrs = op["attrs"]
    groups = int(attrs.get("group", 1) or 1)
    kernel = _pair(attrs.get("kernel_shape") or shape[2:4], (1, 1))
    stride = _pair(attrs.get("strides"), (1, 1))
    params = {
        "in_channels": int(shape[1]) * groups,
        "out_channels": int(shape[0]),
        "kernel_size": _size(kernel),
        "stride": _size(stride),
        "padding": _padding(attrs),
        "groups": groups,
    }
    return Stage(_name(op, index), "conv2d", params)


def _flatten(op: Dict[str, Any], index: int) -> Stage:
    """Map a ``Flatten`` node to a ``flatten`` stage."""
    axis = int(op["attrs"].get("axis", 1) or 1)
    return Stage(_name(op, index), "flatten", {"start_dim": axis})


def _pool(op: Dict[str, Any], index: int) -> Stage:
    """Map an ``AveragePool`` node to an ``avgpool2d`` stage."""
    attrs = op["attrs"]
    kernel = _pair(attrs.get("kernel_shape"), (1, 1))
    stride = _pair(attrs.get("strides"), kernel)
    params = {
        "kernel_size": _size(kernel),
        "stride": _size(stride),
        "padding": _padding(attrs),
    }
    return Stage(_name(op, index), "avgpool2d", params)


def _dropout(op: Dict[str, Any], index: int) -> Stage:
    """Map a ``Dropout`` node to the identity ``dropout`` stage."""
    return Stage(_name(op, index), "dropout", {})


#: ONNX op type -> the builder that faithfully renders it.
OP_BUILDERS: Dict[str, OpBuilder] = {
    "Gemm": _linear,
    "MatMul": _linear,
    "Conv": _conv,
    "Flatten": _flatten,
    "AveragePool": _pool,
    "Dropout": _dropout,
}


def _stage_for(op: Dict[str, Any], index: int) -> Optional[Stage]:
    """Return the stage for ``op``, ``None`` for a passthrough, else raise."""
    builder = OP_BUILDERS.get(op["op_type"])
    if builder is not None:
        return builder(op, index)
    if op["op_type"] in PASSTHROUGH:
        return None
    raise UnsupportedOnnxOpError(
        op["op_type"], "no stage kind represents it"
    )


def _dedupe(stages: List[Stage]) -> List[Stage]:
    """Return ``stages`` with any duplicate names made unique."""
    seen: set = set()
    unique: List[Stage] = []
    for index, stage in enumerate(stages):
        name = stage.name
        if name in seen:
            name = f"{name}_{index}"
        seen.add(name)
        unique.append(Stage(name, stage.kind, dict(stage.params)))
    return unique


def _map_ops(ops: List[Dict[str, Any]]) -> List[Stage]:
    """Return the stages for every mappable op, in graph order."""
    stages: List[Stage] = []
    for index, op in enumerate(ops):
        stage = _stage_for(op, index)
        if stage is not None:
            stages.append(stage)
    return stages


def spec_from_ops(model: Any) -> TopologySpec:
    """Map an ONNX ``model``'s ops to a chained topology spec.

    Raises :class:`UnsupportedOnnxOpError` naming the first op with no
    faithful stage, so an unmappable graph is refused rather than truncated.
    """
    stages = _map_ops(api.graph_ops(model))
    if not stages:
        raise UnsupportedOnnxOpError("graph", "no ops map to a stage")
    return chain(_dedupe(stages))


def _spec_of(model: Any) -> Tuple[TopologySpec, str]:
    """Return ``(spec, source)`` from metadata, else the op mapping."""
    values = api.metadata_values(model)
    recorded = metadata.decode(values)
    if recorded is not None:
        return recorded, "metadata"
    return spec_from_ops(model), "ops"


def load_spec(path: str) -> Tuple[TopologySpec, str]:
    """Load the ONNX file at ``path`` and return ``(spec, source)``."""
    return _spec_of(api.load_onnx(path))


def spec_from_file(path: str) -> TopologySpec:
    """Return just the topology spec imported from ``path``."""
    return load_spec(path)[0]


def import_report(path: str) -> Dict[str, Any]:
    """Return a JSON-able report of importing the model at ``path``."""
    model = api.load_onnx(path)
    spec, source = _spec_of(model)
    return {
        "path": path,
        "source": source,
        "topology": metadata.topology_of(api.metadata_values(model)),
        "ops": api.op_types(model),
        "stages": [
            {"name": stage.name, "kind": stage.kind} for stage in spec.stages
        ],
        "edges": [[edge.source, edge.target] for edge in spec.edges],
    }
